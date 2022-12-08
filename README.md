# CycleGAN and Denoising Techniques To Convert Images into Monet Paintings
In this project, we provide a TensorFlow implementation of a CycleGAN along with various Python scripts that perform various filters on images including a local means filter, local median filter, non-local means filter, and a histogram equalization.

### About Our Project
Our project is based on the Kaggle competition “I’m Something of a Painter Myself” which uses generative adversarial networks (GAN) to generate images that appear as they were painted by Monet. Specifically, the Cycle-Consistent Adversarial Network (CycleGAN) architecture, which comprises two separate generator and discriminator models each, was implemented and trained to generate Monet paintings based on a data set of unpaired Monet paintings and real images. To test our accuracy, we trained a binary classifier to identify Monet paintings from other images, and used the classifier on our generated images to measure the quality of our model. To further improve our images, techniques such as local mean, local median, and non local mean filters were applied to the generated images and the results were compared and analyzed. The first two techniques generated the strongest results.

### Implementation
We first implemented the Cycle GAN to be able to generate Monet style images. We train it with a Binary Loss Entropy loss function on the cycle loss as described above. The discriminator includes 4 convolution layers, and the generator follows a standard encoder decoder network involving multiple convolution layers. We train using gradient descent in the normal fashion. We trained using 50 epochs which took around 4 hours on a laptop. We trained using Monet2Photo dataset containing 1193 Monet paintings and 7038 photos. After the GAN was trained, we ran it on the given Kaggle dataset of 7038 photos to get the final predictions.

From there, we decided that we wanted to improve the images generated by the CycleGAN. In order to facilitate this, we first trained another classifier. In essence, we created another discriminator, but in order to reduce noise from the GAN and train on the same dataset, we decided to train our classifier separately. We trained a default tensorflow image classifier which takes an input and outputs probabilities for two classes, image and Monet. We trained using a random subset of 300 images on both to avoid the problems relating to unbalanced data. After training the classifier, we implemented a variety of algorithms including local means/medians, non local means, and histogram equalization. We used standard packages from OpenCV and SciPy to facilitate this. We again ran the classifier on these images and reported back the classification results. We do not combine methods, but change one variable each time we run.

<img width="1095" alt="Screenshot 2022-12-08 at 5 33 01 PM" src="https://user-images.githubusercontent.com/56417955/206581229-68afd41d-7398-4889-9ac1-5b211979f2c7.png">

<img width="558" alt="Screenshot 2022-12-08 at 5 34 23 PM" src="https://user-images.githubusercontent.com/56417955/206581462-15e9904b-e7ea-4b8b-8b8d-eab99ba6a5bd.png">
