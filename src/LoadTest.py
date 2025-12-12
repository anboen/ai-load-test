import os
import time
import random
import json

from threading import Thread
from pathlib import Path

from openai import OpenAI
from pandas import DataFrame
from dotenv import load_dotenv

from abc import ABC, abstractmethod


class Configs:
    def __init__(self, env_file: str,
                 general_file: str = "general.env",
                 base_path: str = "./envs"):

        base_path_obj = Path(base_path)
        load_dotenv(base_path_obj / env_file)
        load_dotenv(base_path_obj / general_file)

        self.load_configs()
        self.log_configs()

    def log_configs(self):
        print("Configurations:")
        print(f"API Key: {self.api_key}")
        print(f"API Base: {self.api_base}")
        print(f"Model: {self.model}")
        print(f"Input Path: {self.input_path}")

    def load_configs(self):
        self.api_key: str = os.getenv("API_KEY", "key")
        self.api_base: str = os.getenv("API_BASE", "https://localhost/v1")
        self.outpath: str = os.getenv("OUTPATH", "./results/")
        self.model: str = os.getenv("MODEL", "mistral-7b-instruct-v0.1")
        self.input_path: str = os.getenv("INPUT_PATH", "./audios")
        self.num_threads: int = int(os.getenv("NUM_THREADS", "5"))
        self.outfile: str = os.getenv("OUTFILE", "results.csv")
        self.system_prompt: str = os.getenv("SYSTEM_PROMPT",
                                            "You are a helpful assistant.")
        self.input_mode: str = os.getenv("INPUT_MODE", "audio")


class Logger:
    def __init__(self):
        self.times = []
        self.models = []
        self.messages = []
        self.responses = []

    def log(self, time: str, model: str, response: str):
        self.times.append(time)
        self.models.append(model)
        self.responses.append(response)

    def save(self, filepath: str):
        df = DataFrame({
            "time": self.times,
            "model": self.models,
            "response": self.responses
        })
        df.index.name = 'position'
        df.to_csv(filepath, index=True, sep=';')


class OpenAIClient (ABC):

    def __init__(self, configs: Configs):
        self.client: OpenAI = self._create_client(configs.api_key,
                                                  configs.api_base)
        self.check_model_availability(self.client, configs.model)

    def _create_client(self, api_key: str, api_base: str) -> OpenAI:
        client = OpenAI(
            api_key=api_key,
            base_url=api_base,
        )
        return client

    def check_model_availability(self, client, model_id):
        models = client.models.list()
        available_model_ids = [model.id for model in models.data]
        print("Available models:", available_model_ids)
        if model_id in available_model_ids:
            print(f"Model '{model_id}' is available.")
        else:
            raise ValueError(f"Model '{model_id}' is not available in VLLM")


class LoadTest (ABC):

    def __init__(self, client: OpenAI, configs: Configs, logger: Logger):
        self.client: OpenAI = client
        self.model = configs.model
        self.input_path = configs.input_path
        self.num_threads = configs.num_threads
        self.system_prompt = configs.system_prompt
        self.logger = logger
        self.prepare()

    def run(self):
        if self.inputs is None or len(self.inputs) == 0:
            raise ValueError("No inputs prepared for load test.")

        random.shuffle(self.inputs)

        threads = []
        if self.num_threads > len(self.inputs):
            raise ValueError("Number of threads exceeds number of inputs .")
        for i in range(self.num_threads):
            threads.append(Thread(target=self.call, args=(self.inputs[i],
                                                          self.logger, i)))
        for thread in threads:
            thread.start()
        for i, thread in enumerate(threads):
            thread.join()
            print(f'Thread {i}/{self.num_threads} has finished.')

    def prepare(self):
        self.inputs = []

    def call(self, request, logger: Logger, n: int):
        print(f'Starting Thread {n}')
        start = time.time()
        response = self._call_server(request)
        logger.log(time=str(time.time() - start),
                   model=self.model,
                   response=response)
        print(f'Finished Thread {n}')

    @abstractmethod
    def _call_server(self, request) -> str:
        pass


class TextLoadTest(LoadTest):
    def _call_server(self, request):
        return self.client.responses.create(
            model=self.model,
            instructions=request["instructions"],
            input=request["input"],
            temperature=request["temperature"]
        ).output_text

    def _prepare_server(self):
        print("Preparing server...")
        start = time.time()
        self.client.responses.create(
            model=self.model,
            instructions="Sag Hallo",
            input="Hallo",
            temperature=0.0
        )
        print(f'Server prepared in {time.time() - start} seconds.')

    def prepare(self):
        self._prepare_server()
        self.inputs = []
        input_path_obj = Path(self.input_path)

        with open(input_path_obj, "r", encoding="utf-8") as f:
            lines = f.readlines()
            print(f"Loaded {len(lines)} lines from {self.input_path}")
            for line in lines:
                request = {
                    "instructions": self.system_prompt,
                    "input": line,
                    "temperature": 0.0
                }
                self.inputs.append(request)


class AudioLoadTest(LoadTest):

    def _call_server(self, request):
        response_str = self.client.audio.transcriptions.create(
            model=self.model,
            file=request["file"],
            response_format="text",
            language="de"
        )
        json_response = json.loads(response_str)
        return json_response["text"]

    def _prepare_server(self):
        print("Preparing server...")
        start = time.time()
        self._call_server(self.inputs[0])
        print(f'Server prepared in {time.time() - start} seconds.')

    def prepare(self):
        self.inputs = []
        input_path_obj = Path(self.input_path)
        for i in range(200):
            files = input_path_obj.glob('*.*')
            for file in files:
                audio_file = open(str(file), "rb")
                request = {
                    "file": audio_file,
                    "response_format": "text"
                }
                self.inputs.append(request)
        print(f"Prepared {len(self.inputs)} audio inputs.")
        self._prepare_server()
