## BeeMitesTensorFlow - Picture Recognition System

Hello there! I'm Lei. Welcome to **BeeMitesTensorFlow**, a picture recognition system dedicated to the identification of bee species and their related mites. This template provides a comprehensive guide for TensorFlow-based object classification, focusing on the bee and mite domain.

### Code Structure

Our classification is achieved through training two robust models using TensorFlow. Here's a glance at the project's directory structure:

- **images/**: 
  - Hosts a variety of images, from bee species to their mites. Includes both test images and GUI-based visuals.
  
- **models/**:
  - Contains the CNN model for bee species and the MobileNet model for mite identification.
  
- **results/**:
  - Provides visual and textual insights from the training. Contains .txt files with training details and graphs for accuracy & loss curves.
  
- **utils/**:
  - Additional scripts and files for project development and personal testing.
  
- **scrape_images.py**:
  - Script to fetch bee and mite images from Bing.
  
- **interface.py**:
  - GUI developed using PyQt5 for uploading and identifying bee and mite species.
  
- **validate_models.py**:
  - Verifies the accuracy of the models against a validation dataset.
  
- **train_bee_cnn.py**:
  - Trains the CNN model for bee species.
  
- **train_mite_mobilenet.py**:
  - Trains the MobileNet model for mite identification.
  
- **requirements.txt**:
  - Lists project dependencies.

Thank you for choosing **BeeMitesTensorFlow**. Dive into the world of bees and their mites! Happy identifying!
