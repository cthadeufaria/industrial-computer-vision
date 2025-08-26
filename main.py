from dataset import ScrewsDataset
from model import FasterRCNNPipeline, openCVPipeline


def main():
    dataset = ScrewsDataset()
    
    opencv_pipeline = openCVPipeline("output_1", dataset)
    opencv_pipeline.run()

    CNN_pipeline = FasterRCNNPipeline(output_dir="output_2", dataset=dataset)
    CNN_pipeline.run()

if __name__ == "__main__":
    main()