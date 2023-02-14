
# Classification of Diffusion Model Generated VS Original Art

As part of this project we have built a CNN model, using transfer learning approach , that can distinguish between AI generated art (created using AI-powered tool Dall.E) and original art created by artists. 

## Problem:

Identify if the image in question is an original artwork or if it was generated by Dall.E. 

## Context: 

With the rise of AI-powered tools, we are faced with the ethical concern of the originality of a creative endeavor. Additionally, AI generated media could potentially cause discourse if it refers to polarizing topics, which we have seen play out with exceptionally realistic deepfakes. Dall.E uses texutual prompts to generate images, using its own extensive library of original art created by human artists, which begs the questions-" who really is the owner of the new art?","are the original artists credited for the new art created using their pieces?","what happens to the creatives if the AI takes over and does their job?","Will there be a loss of appreciation for the artists around us?". Along with this AI has also been used to generate fraudulent art and with the rise of investments in NFTs, to determine the authenticity of a digital piece, we find ourselves with a very opportune usecase where this project is useful.

## Impact:

With this project we plan to help differentiate between an original artwork and AI generated art which would help people and companies wanting to own a piece of art, know about its authenticity. An extrapolation of this project is detecting deepfakes in images as well as video frames.

## Approach:

For this we created a custom dataset where we used the Dall.E api to generate 600+ images using varying random prompts. For the Original artwork we programatically pulled the images (600+) using a list of urls from The Paintings Dataset (https://www.robots.ox.ac.uk/~vgg/data/paintings/ ) curated by Elliot J. Crowley, Ernesto Coto and Andrew Zisserman.
For modelling we used a pre-trained Resnet-18 model for feature extraction and fine tuned it with the data that we gathered.
We decided to use Accuracy as as metric as our dataset was no imbalanced and the cost of the prediction is not severe (as per the usecase and the stakes defined we may pivot to using some other metric to determine how the model is doing or for hyper parameter tuning). We did however capture the confusion matrix metrics for every model we ran.
We wanted to expose a public api and make this app accessible to anyone who wants to try it but doesnt care to run any scripts. That is why we hosted our app on Hugging Face and deployed it using gradio for easy UI navigation. 

## Limitations:

Since the model was trained on images generated using Dall.E, it is able to weed out those AI- generated images better as it learns the patterns/signature that Dall.E uses. We would like to enhance this model by feeding it images from a wide variety of AI-powered image generation tools to generalize better

## User guide:

User can access the app through the link-https://huggingface.co/spaces/NehaBardeDUKE/ComputerVision_540 . This takes the user to a public space on hugging face where they can drop an image in the input box as prompted and they will get the prediction along with the probability of the prediction. We chose to add this in order to provide the user with explanability of the result.
Demo :https://user-images.githubusercontent.com/110474064/218609768-ea63be4d-7f1c-42ff-a4c2-e99ddb15f909.mp4

## Evaluation Results:

### Deep learning Model
We chose to implement a pre-trained resnet 18 model and fine tuned it to our data. We created a total of 4 models by tuning the hyperparameters and then unfreezing all the training layers and performing further hyper parameter tuning. Below is the result we saw for each case on the test dataset-
Model 1: epoch num: 10; lr_scheduler step_size = 7; lr = 0.001; gamma = 0.1; batch_size = 10
![image](https://user-images.githubusercontent.com/110474064/218646451-2a924de7-cea3-41e2-b43c-0bd400404fca.png)
Model 2: epoch num: 25; lr_scheduler step_size = 8; lr = 0.001; gamma = 0.1; batch_size = 10
![image](https://user-images.githubusercontent.com/110474064/218646668-3d5df486-d16b-4a0d-b006-f19ae83014c4.png)
Model 3: epoch num: 15; lr_scheduler step_size = 7; lr = 0.001; gamma = 0.1; batch_size = 10; unfreeze parameter training w/ 12 epochs
![image](https://user-images.githubusercontent.com/110474064/218646904-d37decdb-8cf9-4de8-9403-806a487de1ba.png)
Model 4: epoch num: 10; lr_scheduler step_size = 7; lr = 0.001; gamma = 0.2; batch_size = 10; unfreeze parameter training w/ 5 epochs
![image](https://user-images.githubusercontent.com/110474064/218647355-5075d3cb-a506-475b-b12f-5a176951880b.png)

While we expected to see the the accuracy increase when we unfroze all the layers and trained the model on our data, we saw that the accuracy actually went down for those test cases. These models were not able to generalize better on the orignal human art. The model 2 also tended to overfit the data and perform poorly with the original art detection. The first model however was the most promising one with a high accuracy and better generalization. In the future we would want to increase our dataset and train on equal high volume artificially generated and original art


### Non-Deep learning Model

We chose to implement the classification of the images using 4 classical ML models - SVM, Decision Trees, Random Forest and KNN. Since the calculations for all of these are fairly computationally intensive (vectorized but the data needs a lot of ram which is why we saw out of memory error), we had to decrease the resolution of the images to 240P and have a fairly small dataset (tradeoff between the feature set and size of data). Even with this, we saw that Random Forest Decision Tree came the closest in terms of accuracy to the deep learning model's result. This is in contrast to the assumption that a clustering algorithm would show a better result since it logically carries that it would create clusters out of similar data points. However clustering (a simple KNN) performed underwhelmingly. 
![image](https://user-images.githubusercontent.com/110474064/218643467-7bd3211b-e0a2-44f8-a0ff-1d290212c422.png)


## References
Matthew Maybe.2020. "Can an AI learn to identify “AI art”?"-https://medium.com/@matthewmaybe/can-an-ai-learn-to-identify-ai-art-545d9d6af226 

Electra Nanou.2023. "The Ethical Pros and Cons of AI Art Generation" -https://www.makeuseof.com/ai-art-generation-ethical-pros-cons/#:~:text=The%20Ethical%20Downsides%20of%20Using,are%20real%20art%20at%20all.

Melissa Heikkilä.2022. "The Algorithm: AI-generated art raises tricky questions about ethics, copyright, and security"- https://www.technologyreview.com/2022/09/20/1059792/the-algorithm-ai-generated-art-raises-tricky-questions-about-ethics-copyright-and-security/

Riccardo Corvi,et.al.2022."On the detection of synthetic images generated by diffusion models"-https://arxiv.org/pdf/2211.00680.pdf

Elliot J.Crowley,Ernesto Coto,Andrew Zisserman,Irina Reshodko.2021.The Paintings Dataset -https://www.robots.ox.ac.uk/~vgg/data/paintings/









