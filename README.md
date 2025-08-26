# industrial-computer-vision


Desafio 1:

Objective:
The objective of this test is to evaluate your ability to design and develop a simple computer vision application to detect and isolate a single industrial component (screw) in a set of images. Each image contains one screw placed on a controlled background, with some variation in orientation, position, and lighting, which simulates the controlled environment of industrial applications.
The main goal here is to observe how you approach the problem, structure your code, and handle practical aspects of computer vision. This task can be performed with the application of traditional computer vision methods or, if you prefer, more advanced methods, such as deep learning.
You are free to design your own solution using the tools you are comfortable with. However, please keep the implementation simple and ensure your code is executable in a standard local environment (e.g., Python, C++, or C#). Furthermore, avoid the use of external platforms, such as AWS, Google Cloud, or other services that would prevent us from evaluating your solution.

Dataset:
• Composed of 320 images, each with a resolution of 1024 x 1024.
• Each image contains a single screw on a controlled background, covering different orientations and positions.
• Ground truth labels are not provided.
• All images are present in the annex folder data.


Evaluation Criteria:
• Detection Accuracy: The solution accurately detects and locates the screw in the images.
• Code Quality: Code is clear, structured and documented.
• Innovative Solutions: Creativity and innovation in approach and problem-solving.
• Documentation: Clarity and comprehensiveness of the development process documentation.

Deliverables:
• Submit all code developed for this challenge.
• Provide a written or schematic document detailing your thought process and development approach.
• Instructions for setting up and executing the solution in a standard local environment.
• Submit example results demonstrating the screw detection and isolation on the provided dataset.

Notes:
• Solutions should remain simple and executable locally without requiring specialized hardware or cloud services.
• The application of environments such as Anaconda, notebooks (Jupyter), or Google Colab is allowed.
• Participants are free to choose their preferred approach, if the solution meets the requirements.


# Solution Documentation

Steps:

0. create and activate a Python virtual environment and install the dependencies on requirements.txt with:
    ``` 
    python3 -m venv .venv 
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

1. Solution 1:
    - Using openCV to manipulate images and create ROI for each bolt image
        - describe steps and logic
    
2. Solution 2:
    - Using pre-trained classification and localization model

3. Solution 3:
    - Using labeles images from 'Solution 1' to create a dataset for fine-tuning a localization model (also segmentation model?)

