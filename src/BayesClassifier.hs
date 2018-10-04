{-# LANGUAGE FlexibleContexts  #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications  #-}
{-# LANGUAGE TypeFamilies      #-}
{- |
Copyright:    Guilherme Grochau Azzi 2018
License:      BSD3

This module implements a naive Bayes classifier for values containing textual
tokens, assigning labels to such values.

We assume that the probability of assigning a particular label is dependent on
the probability of certain tokens occuring.  On the other hand, we make the
naive assumption that the probability of each token is independent on the other
tokens.  Although this is certainly false, it still produces good results with a
simple algorithm.
-}

module BayesClassifier
  (
    -- * Tokens

    -- | Tokens are just chunks of text that are considered to be meaningful for
    -- the attribution of labels.

    -- $tokenization
    Token
  , Tokenizable(..)
  , token
  , tokText

  -- * Bayesian Classifier
  , Classifier
  , emptyClassifier
  , Labeled(..)
  -- ** Learning
  , learn
  , buildClassifier
  -- ** Classifying
  , classify, classifyToks
  , classify', classifyToks'
  ) where

import           Control.Applicative
import           Control.Arrow
import           Control.Monad
import qualified Data.Char            as Char
import           Data.Foldable
import qualified Data.List            as List
import           Data.Map.Strict      (Map)
import qualified Data.Map.Strict      as Map
import           Data.Maybe           (fromMaybe)
import qualified Data.Ord             as Ord
import           Data.Text            (Text)
import qualified Data.Text            as Text
import           Data.Time.Calendar   (fromGregorian)
import           Data.Time.Format     (defaultTimeLocale, parseTimeM)
import           Hledger              (AccountName, Posting (..),
                                       Transaction (..))
import qualified Text.Megaparsec      as Parser
import qualified Text.Megaparsec.Char as Parser


newtype Token = Token
  { tokText :: Text -- ^ Obtain the textual representation of a token.
  } deriving (Eq, Show, Ord)

-- | Obtain the token that corresponds to the given text.
--
-- This is the inverse of 'tokText', that is, @tokText . token = id@ and
-- @token . tokText = id@.
token :: Text -> Token
token = Token . Text.toCaseFold

class Tokenizable a where
  -- | Extract the tokens that occur in the given value.  The same token may
  -- appear multiple times.
  tokenize :: a -> [Token]

-- | A naive Bayes classifier that assigns @label@s to multisets of 'Token's.
data Classifier label = Model
  { nEntries              :: Int
  , nTotalTokens          :: Int
  , nTokenOccurs          :: Map Token Int
  , nTokenOccursWithLabel :: Map (Token, label) Int
  , nEntriesWithLabel     :: Map label Int
  , nTokensWithLabel      :: Map label Int
  } deriving (Show)

nDistinctTokens :: Classifier label -> Int
nDistinctTokens = Map.size . nTokenOccurs

-- Text is tokenized according to the rules described later.
instance Tokenizable Text where
  tokenize text =
    fromMaybe (error "Internal error tokenizing text.")
    $ Parser.parseMaybe tokenizer text

-- Strings are tokenized exactly like Text.
instance {-# OVERLAPS #-} Tokenizable String where
  tokenize = tokenize . Text.pack

-- | Lists are tokenized by concatenating the tokens of each item.
instance {-# OVERLAPPABLE #-} Tokenizable a => Tokenizable [a] where
  tokenize = concatMap tokenize

-- | Lists of tokens are trivially tokenizable.
instance {-# OVERLAPS #-} Tokenizable [Token] where
  tokenize = id

instance Tokenizable Transaction where
  tokenize t = concat $ tokenize (tdescription t) : tokenize (tcomment t) : tokenize (tpostings t) : map tokenizeTag (ttags t)

instance Tokenizable Posting where
  tokenize p = concat $ [token (paccount p)] : tokenize (pcomment p) : map tokenizeTag (ptags p)

tokenizeTag :: Tokenizable a => (Text, a) -> [Token]
tokenizeTag (name, val) = token name : tokenize val

-- | Type class for values that are labeled.  Each value may be assigned
-- multiple labels.
class Labeled a where
  type Label a
  labelsOf :: a -> [Label a]

instance Labeled Transaction where
  type Label Transaction = AccountName
  labelsOf = map paccount . tpostings

-- | An empty, untrained classifier.
emptyClassifier :: Classifier label
emptyClassifier = Model 0 0 Map.empty Map.empty Map.empty Map.empty

-- | Build a classifier from the given values.
buildClassifier :: (Tokenizable a, Labeled a, Ord (Label a)) =>
                  [a] -> Classifier (Label a)
buildClassifier = foldl' (flip learn) emptyClassifier

-- | Train a classifier on the given value.
learn :: (Tokenizable a, Labeled a, Ord (Label a)) =>
         a -> Classifier (Label a) -> Classifier (Label a)
learn entry c =
  let c' = c { nEntries = nEntries c + 1
             , nTotalTokens = nTotalTokens c + nEntryTokens }
  in List.foldl' learnFromLabel c' (labelsOf entry)
  where
    entryTokens = tokenize entry
    nEntryTokens = length entryTokens

    learnFromLabel c lbl =
      let c' = c { nEntriesWithLabel = incrementAt lbl 1 (nEntriesWithLabel c)
                 , nTokensWithLabel = incrementAt lbl nEntryTokens (nTokensWithLabel c) }
      in List.foldl' (learnFromLabelAndToken lbl) c' entryTokens

    learnFromLabelAndToken lbl c tok =
      c { nTokenOccurs = incrementAt tok 1 (nTokenOccurs c)
        , nTokenOccursWithLabel = incrementAt (tok, lbl) 1 (nTokenOccursWithLabel c) }

incrementAt :: (Ord k, Num a) => k -> a -> Map k a -> Map k a
incrementAt k v = Map.alter (Just . maybe v (+v)) k

-- | Estimate the most likely applicable label for the given list of tokens.
--  The first argument is a predicate determining which labels are applicable.
classifyToks :: (Ord label) =>
                (label -> Bool) -> [Token] -> Classifier label -> label
classifyToks labelApplicable toks = fst . head
  . classifyToks' @Double labelApplicable toks

-- | Estimate the likelihood that each known applicable label is assigned to the
-- given list of tokens, listing them with descending likelihood.  The first
-- argument is a predicate determining which labels are applicable.
classifyToks' :: (Ord prob, Fractional prob, Ord label) =>
                 (label -> Bool) -> [Token] -> Classifier label -> [(label, prob)]
classifyToks' labelApplicable toks model =
  List.sortOn (Ord.Down . snd)
  . map (id &&& pLabelGivenToks)
  . filter labelApplicable
  $ knownLabels model
  where
    knownLabels = Map.keys . nTokensWithLabel
    containsToken tok model = tok `Map.member` nTokenOccurs model

    pLabelGivenToks lbl = probTokensAndLabel model toks' lbl / pToks
    pToks = probTokens model toks'
    toks' = filter (`containsToken` model) toks

-- | Estimate the most likely applicable label for the given tokenizable value.
-- The first argument is a predicate determining which labels are applicable.
classify :: (Tokenizable a, Ord label) =>
            (label -> Bool) -> a -> Classifier label -> label
classify labelApplicable = classifyToks labelApplicable . tokenize

-- | Estimate the likelihood that each known applicable label is assigned to the
-- given tokenizable value, listing them with descending likelihood.  The first
-- argument is a predicate determining which labels are applicable.
classify' :: (Tokenizable a, Ord label, Ord prob, Fractional prob) =>
             (label -> Bool) -> a -> Classifier label -> [(label, prob)]
classify' labelApplicable = classifyToks' labelApplicable . tokenize

-- | @probTokensAndLabel model toks lbl = P(lbl | toks)@
probTokensAndLabel :: (Fractional prob, Ord label) =>
                      Classifier label -> [Token] -> label -> prob
probTokensAndLabel model toks lbl =
  probTokensGivenLabel model toks lbl * probLabel model lbl

-- | @probTokens model toks = P(toks)@
probTokens :: (Fractional prob, Ord label) =>
              Classifier label -> [Token] -> prob
probTokens model toks =
  product [ probToken model tok | tok <- toks ]

-- | @probTokensGivenLabel model toks lbl = P(toks | lbl)@
probTokensGivenLabel :: (Fractional prob, Ord label) =>
                        Classifier label -> [Token] -> label -> prob
probTokensGivenLabel model toks lbl =
  product [ probTokenGivenLabel model tok lbl | tok <- toks ]

-- | @probTokenGivenLabel model tok lbl = P(tok | lbl)@
probTokenGivenLabel :: (Fractional prob, Ord label) =>
                       Classifier label -> Token -> label -> prob
probTokenGivenLabel model tok lbl =
  (Map.findWithDefault 0 (tok, lbl) (nTokenOccursWithLabel model) + 1)
  // (Map.findWithDefault 0 lbl (nTokensWithLabel model) + nDistinctTokens model)

-- | @probToken model tok = P(tok)@
probToken :: (Fractional prob) => Classifier label -> Token -> prob
probToken model tok =
  Map.findWithDefault 0 tok (nTokenOccurs model)
  // nTotalTokens model

-- | @probLabel model lbl = P(lbl)@
probLabel :: (Fractional prob, Ord label) => Classifier label -> label -> prob
probLabel model lbl =
  Map.findWithDefault 0 lbl (nEntriesWithLabel model)
  // nEntries model

(//) :: (Fractional r, Integral m, Integral n) => m -> n -> r
a // b = fromIntegral a / fromIntegral b


{- $tokenization

'String's and 'Text' are tokenized following a relatively simple strategy.  We
identify three kinds of tokens.

  1. Date tokens: we try to identify dates and replace the particular values
  with the inferred format.  For example, "2018-01-10" is replaced with
  "%Y-%m-%d".  The idea behind this is that the date format is more directly
  linked to labels than the particular date, and was taken from
  <https://tomszilagyi.github.io/payment-matching/>.

  2. Numeric tokens: this includes unsigned integers and decimals.

  3. Textual tokens: any other sequence of characters that doesn't include
  spaces or punctuation, as defined by the unicode standard.
-}

type TextParser = Parser.Parsec () Text

-- tokenizer :: Parser [Token]
tokenizer :: TextParser [Token]
tokenizer = seps *> many (tok <* seps)
  where
    tok = textToken <|> Parser.try dateToken <|> numberToken
    seps = Parser.takeWhileP Nothing isSeparator

isSeparator :: Char -> Bool
isSeparator c = Char.isSpace c || Char.isPunctuation c

textToken, numberToken, dateToken :: TextParser Token
textToken = token <$> Parser.takeWhile1P Nothing (\c -> not $ isSeparator c || Char.isDigit c)
numberToken = token <$> number
  where
    number =
      (<>) <$> Parser.takeWhile1P Nothing Char.isDigit <*> (Parser.try fracPart <|> pure "")
    fracPart =
      Text.cons <$> Parser.satisfy (\c -> c==',' || c=='.') <*> Parser.takeWhile1P Nothing Char.isDigit

dateToken = token . Text.pack <$> degradedDate
  where
    degradedDate = Parser.choice $ map Parser.try [d8, d6, d4d2d2, d2d2d4, d2d2d2]

    d8 = do
      digits <- Parser.count 8 Parser.digitChar
      Parser.notFollowedBy Parser.digitChar
      tryFormats digits ["%Y%m%d", "%d%m%0Y", "%m%d%Y"]

    d6 = do
      digits <- Parser.count 6 Parser.digitChar
      Parser.notFollowedBy Parser.digitChar
      tryFormats digits ["%y%m%d", "%d%m%y", "%m%d%y"]

    d4d2d2 = digitsSep 4 2 2 [["%Y","%m","%d"]]
    d2d2d4 = digitsSep 2 2 4 [["%d","%m","%Y"],["%m","%d","%Y"]]
    d2d2d2 = digitsSep 2 2 2 [["%y","%m","%d"],["%d","%m","%y"],["%m","%d","%y"]]

    digitsSep n1 n2 n3 formats = do
      digits1 <- Parser.count n1 Parser.digitChar
      sep <- Parser.satisfy isDateSep
      digits2 <- Parser.count n2 Parser.digitChar
      _ <- Parser.char sep
      digits3 <- Parser.count n3 Parser.digitChar
      Parser.notFollowedBy Parser.digitChar
      let withSep = List.intercalate [sep]
          text = withSep [digits1, digits2, digits3]
      tryFormats text (map withSep formats)

    isDateSep c = c == '-' || c == '/' || c == '.'
    tryFormats text formats = Parser.choice $ map (Parser.try . parseDate text) formats
    parseDate text format = do
      date <- parseTimeM False defaultTimeLocale format text
      guard (withinReasonableRange date)
      return format
    withinReasonableRange date =
      fromGregorian 2000 1 1 <= date && date <= fromGregorian 2099 12 31
