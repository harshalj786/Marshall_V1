from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader,WebBaseLoader,SeleniumURLLoader,CSVLoader
from langchain_core.runnables import Runnable
from langchain_core.documents import Document
from typing import Literal,Optional
from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from pydantic import BaseModel,Field

class LoadDocs(Runnable):
    def __init__(self,file_path,mode : Optional[Literal["pdf","scanned_pdf","web_url","txt","youtube","folder","csv"]] = "pdf"):
        #file_path is video ID for mode == youtube
        #If mode = "folder" then you should give glob_pattern while calling invoke function
        #This is not for .md file
        #Give glob_pattern only when mode = "folder"
        self.file_path = file_path
        self.mode = mode
    def invoke(self,input,config = None):
        glob_pattern :Optional[str] = input.get("glob_pattern")
        if self.mode == "pdf":
         pdf_loader = PyPDFLoader(file_path=self.file_path)
         docs = pdf_loader.load()
         return docs
        elif self.mode == "youtube":
           try:
              ytt_api = YouTubeTranscriptApi()
              transcript_list = ytt_api.fetch(self.file_path)
              transcript = " ".join([t.text for t in transcript_list])
              return [Document(page_content=transcript)]
           except TranscriptsDisabled:
              raise ValueError(f"Transcripts are disabled for {self.file_path}")
        elif self.mode == "scanned_pdf":
           pdf_loader = PDFPlumberLoader(
              file_path=self.file_path
           )
           docs = pdf_loader.load()
           return docs
        elif self.mode == "web_url":
           url_loader = WebBaseLoader(self.file_path)
           docs = url_loader.load()
           return docs
        elif self.mode == "csv":
           return (CSVLoader(file_path=self.file_path).load())
        elif self.mode == "txt":
           txt_loader = TextLoader(file_path=self.file_path)
           docs = txt_loader.load()
           return docs
        elif self.mode == "folder":
           if not glob_pattern:
            raise ValueError("glob_pattern is required for folder mode")
           elif glob_pattern.endswith(".pdf"):
             loader = DirectoryLoader(
            path = self.file_path,
            glob = glob_pattern,
            show_progress=True,
            loader_cls=PyPDFLoader,
           )
             docs = loader.load()
             return docs
           elif glob_pattern.endswith(".txt"):
            loader = DirectoryLoader(
                path = self.file_path,
                glob = glob_pattern,
                show_progress=True,
                loader_cls=TextLoader,
            )
            docs = loader.load()
            return docs
           elif glob_pattern.endswith(".csv"):
              loader = DirectoryLoader(
                 path = self.file_path,
                 loader_cls=CSVLoader,
                 glob = glob_pattern,
                 show_progress=True
              )
              return (loader.load())
           else:
            raise ValueError("Unsupported file type in folder mode")