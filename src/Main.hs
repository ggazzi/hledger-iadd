{-# LANGUAGE OverloadedStrings, LambdaCase #-}

module Main where

import           Brick
import           Brick.Widgets.Border
import           Brick.Widgets.Edit
import           Brick.Widgets.List
import           Brick.Widgets.List.Utils
import           Graphics.Vty

import           Control.Lens
import           Control.Monad
import           Control.Monad.IO.Class
import           Control.Monad.Trans.Except
import           Data.Default
import           Data.Maybe
import           Data.Monoid
import           Data.Text (Text)
import qualified Data.Text as T
import           Data.Text.Zipper
import           Data.Time hiding (parseTime)
import qualified Data.Vector as V
import qualified Hledger as HL
import qualified Hledger.Read.JournalReader as HL
import           System.Environment

import           Model
import           View

data AppState = AppState
  { asEditor :: Editor
  , asStep :: Step
  , asJournal :: HL.Journal
  , asContext :: List Text
  , asSuggestion :: Maybe Text
  , asMessage :: Text
  , asFilename :: FilePath
  }


draw :: AppState -> [Widget]
draw as = [ui]
  where ui =  viewState (asStep as)
          <=> hBorder
          <=> (viewQuestion (asStep as)
               <+> viewSuggestion (asSuggestion as)
               <+> txt ": "
               <+> renderEditor (asEditor as))
          <=> hBorder
          <=> expand (viewContext (asContext as))
          <=> hBorder
          <=> txt (asMessage as <> " ") -- TODO Add space only if message is empty

event :: AppState -> Event -> EventM (Next AppState)
event as ev = case ev of
  EvKey KEsc [] -> halt as
  EvKey (KChar 'n') [MCtrl] -> continue as { asContext = listMoveDown $ asContext as
                                           , asMessage = ""}
  EvKey (KChar 'p') [MCtrl] -> continue as { asContext = listMoveUp $ asContext as
                                           , asMessage = ""}
  EvKey (KChar 'c') [MCtrl] -> liftIO (reset as) >>= continue
  EvKey KEnter [MMeta] -> liftIO (doNextStep False as) >>= continue
  EvKey KEnter [] -> liftIO (doNextStep True as) >>= continue
  _ -> setContext <$>
       (AppState <$> handleEvent ev (asEditor as)
                 <*> return (asStep as)
                 <*> return (asJournal as)
                 <*> return (asContext as)
                 <*> return (asSuggestion as)
                 <*> return ""
                 <*> return (asFilename as))
       >>= continue

reset :: AppState -> IO AppState
reset as = do
  sugg <- suggest (asJournal as) DateQuestion
  return as
    { asStep = DateQuestion
    , asEditor = clearEdit (asEditor as)
    , asContext = ctxList V.empty
    , asSuggestion = sugg
    , asMessage = "Transaction aborted"
    }

setContext as = as { asContext = flip listSimpleReplace (asContext as) $ V.fromList $
  context (asJournal as) (editText as) (asStep as) }

editText = T.pack . concat . getEditContents . asEditor

doNextStep useSelected as = do
  let name = fromMaybe (editText as) $
               msum [ if useSelected then snd <$> listSelectedElement (asContext as) else Nothing
                    , asMaybe (editText as)
                    , asSuggestion as
                    ]
  s <- nextStep (asJournal as) name (asStep as)
  case s of
    Left err -> return as { asMessage = err }
    Right (Finished trans) -> do
      liftIO $ addToJournal trans (asFilename as)
      sugg <- suggest (asJournal as) DateQuestion
      return AppState
        { asStep = DateQuestion
        , asJournal = HL.addTransaction trans (asJournal  as)
        , asEditor = clearEdit (asEditor as)
        , asContext = ctxList V.empty
        , asSuggestion = sugg
        , asMessage = "Transaction written to journal file"
        , asFilename = asFilename as
        }
    Right (Step s') -> do
      sugg <- suggest (asJournal as) s'
      let ctx' = ctxList $ V.fromList $ context (asJournal as) "" s'
      return as { asStep = s'
                , asEditor = clearEdit (asEditor as)
                , asContext = ctx'
                , asSuggestion = sugg
                }

asMaybe :: Text -> Maybe Text
asMaybe t
  | T.null t  = Nothing
  | otherwise = Just t

attrs = attrMap defAttr
  [ (listSelectedAttr, black `on` white) ]

clearEdit edit = edit & editContentsL .~ stringZipper [""] (Just 1)

addToJournal :: HL.Transaction -> FilePath -> IO ()
addToJournal trans path = appendFile path (show trans)

ledgerPath home = home <> "/.hledger.journal"

main :: IO ()
main = do
  home <- getEnv "HOME" -- FIXME
  let path = ledgerPath home
  journalContents <- readFile path
  Right journal <- runExceptT $ HL.parseJournalWith HL.journal True path journalContents

  let edit = editor "Edit" (str . concat) (Just 1) ""

  sugg <- suggest journal DateQuestion

  let as = AppState edit DateQuestion journal (ctxList V.empty) sugg "Welcome" path

  void $ defaultMain app as

  where app = App { appDraw = draw
                  , appChooseCursor = showFirstCursor
                  , appHandleEvent = event
                  , appAttrMap = const attrs
                  , appLiftVtyEvent = id
                  , appStartEvent = return
                  } :: App AppState Event

expand = padBottom Max

ctxList v = (if V.null v then id else listMoveTo 0) $ list "Context" v 1
