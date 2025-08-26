from dataset import ScrewsDataset
from model import openCVPipeline


def main():
    dataset = ScrewsDataset()
    pipeline = openCVPipeline("output_1", dataset)
    pipeline.run()

if __name__ == "__main__":
    main()