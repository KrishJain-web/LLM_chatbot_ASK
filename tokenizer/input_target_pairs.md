Unsupervised learning:In this machine learnong algorithm LLM doesn't knows about the output it only sees the input and find a perfect relation
                      between looking over the patterns.
Input target pairs:

  <img width="381" height="40" alt="image" src="https://github.com/user-attachments/assets/0753e36a-d084-4161-9a7a-9474cc56e5bc" />

  not circled text is are the masks that are not seen by the LLM.


Parellel computing : When large data is divide into small parts for better computing.

Now using , pytorch(data loader) for large dataset and Tensor(here as a 2 dimensional array).
so basically the dataset that we provided gets converted into a 2D array due to tensor and that helps to give the next word by just shifting the array one.
**Working of input-target pairing helps us to give the next word from the dataset . It happens by completely tokenizing the dataset and storing it in a tensor 2d matrix and then shifting it by one such that it looks a new word has been generated .

<img width="722" height="37" alt="image" src="https://github.com/user-attachments/assets/aed3e8c3-330a-45d6-a682-8e61053ba6bb" />


example iteration over 4X8

<img width="448" height="477" alt="image" src="https://github.com/user-attachments/assets/f37f8561-8c13-44e1-98ec-e8cadc526fff" />


                      

