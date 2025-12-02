import argparse
from LoadTest import Configs, OpenAIClient, TextLoadTest, AudioLoadTest, Logger
from datetime import datetime
from pathlib import Path


def main(args):

    configs = Configs(args.env_file)
    logger = Logger()
    client = OpenAIClient(configs)
    if configs.input_mode == "text":
        load_test = TextLoadTest(client.client, configs, logger)
    elif configs.input_mode == "audio":
        load_test = AudioLoadTest(client.client, configs, logger)
    else:
        raise ValueError(f"Unsupported INPUT_MODE: {configs.input_mode}")

    load_test.run()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    outpath = Path(configs.outpath) / \
        f"{timestamp}_{args.gpu}_{configs.outfile}"
    print(f"Saving results to {outpath}")
    logger.save(str(outpath))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Load Tests for AI Models")
    parser.add_argument(
        "env_file", help="environment file with configurations")
    parser.add_argument("-g", "--gpu", type=str, default=1,
                        help="GPU model")

    args = parser.parse_args()
    main(args)
