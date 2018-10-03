{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE FlexibleInstances   #-}
{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE TypeApplications    #-}
{-# LANGUAGE TypeFamilies        #-}
{- |
Copyright:    Guilherme Grochau Azzi 2018
License:      BSD3

This module implements a naive Bayes classifier for values from which textual
tokens can be extracted.

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
import           Control.Monad.State
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
import           Lens.Micro
import           Lens.Micro.Mtl
import           Lens.Micro.GHC ()
import           Lens.Micro.TH (makeLenses)
import qualified Text.Megaparsec      as Parser
import qualified Text.Megaparsec.Char as Parser
import Hledger (AccountName, Posting(..), Transaction(..))


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
  { _nEntries              :: Int
  , _nTotalTokens          :: Int
  , _nTokenOccurs          :: Map Token Int
  , _nTokenOccursWithLabel :: Map (Token, label) Int
  , _nEntriesWithLabel     :: Map label Int
  , _nTokensWithLabel      :: Map label Int
  } deriving (Show)
makeLenses ''Classifier

nDistinctTokens :: SimpleGetter (Classifier label) Int
nDistinctTokens = nTokenOccurs . to Map.size

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
learn entry = execState $ do
  let tokens = tokenize entry
      nTokens = length tokens
  nEntries += 1
  nTotalTokens += nTokens
  forM_ (labelsOf entry) $ \lbl -> do
    nEntriesWithLabel . at lbl +?= 1
    nTokensWithLabel . at lbl +?= nTokens
    forM_ tokens $ \tok -> do
      nTokenOccurs . at tok +?= 1
      nTokenOccursWithLabel . at (tok, lbl) +?= 1

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
  let
    toks'= filter (`containsToken` model) toks
    pToks = probTokens model toks'
    pLabelGivenToks lbl = probTokensAndLabel model toks' lbl / pToks
  in
    List.sortOn (Ord.Down . snd)
    $ model ^.. knownLabels
              . filtered labelApplicable
              . to (id &&& pLabelGivenToks)
  where
    knownLabels = nTokensWithLabel . to Map.keys . each
    containsToken tok model = tok `Map.member` (model ^. nTokenOccurs)

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
  (fromMaybe 0 (model ^. nTokenOccursWithLabel . at (tok, lbl)) + 1)
  // (fromMaybe 0 (model ^. nTokensWithLabel . at lbl) + (model ^. nDistinctTokens))

-- | @probToken model tok = P(tok)@
probToken :: (Fractional prob) => Classifier label -> Token -> prob
probToken model tok =
  fromMaybe 0 (model ^. nTokenOccurs . at tok)
  // (model ^. nTotalTokens)

-- | @probLabel model lbl = P(lbl)@
probLabel :: (Fractional prob, Ord label) => Classifier label -> label -> prob
probLabel model lbl =
  fromMaybe 0 (model ^. nEntriesWithLabel . at lbl)
  // (model ^. nEntries)

(//) :: (Fractional r, Integral m, Integral n) => m -> n -> r
a // b = fromIntegral a / fromIntegral b

(+?=) :: (MonadState s m, Num a) => ASetter s s (Maybe a) (Maybe a) -> a -> m ()
l +?= n = modifying l (Just . (+n) . fromMaybe 0)
infix 4 +?=


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
