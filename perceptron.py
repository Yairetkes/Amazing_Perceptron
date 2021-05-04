import numpy as np

data_amount = 500
learning_const = 0.1
with_bias = False
no_of_digits = 21
threshold = 11

class Perceptron(object):
    def __init__(self, data_amount: int, learning_const: int, with_bias: bool, no_of_digits: int, threshold: int):
        self.data_amount = data_amount
        self.learning_const = learning_const
        self.with_bias = with_bias
        self.no_of_digits = no_of_digits
        self.threshold = threshold
    
        self.data_set = np.random.randint(0, 2, size = (self.data_amount, self.no_of_digits + 1))
        # this line creates a 2D array which is the data set for our algorithm to learn.
        #  every row represent 1 binary number.
        # the last cell of every number will contain, later on, it's tagging (y) - 1 if it has more '1' digits or -1 if it has more '0's...

        self.test_data_set = np.random.randint(0, 2, size = (self.data_amount, self.no_of_digits + 1))
        # this line creates another 2D array same size as the previous one, but this one is for later - to test if the perceptron learned well.

        self.weights =  np.array([0]*self.no_of_digits)

    def tagging(self, data_set):#this finction add the tag y - 1 if it has more '1' digits or -1 if it has more '0's
        for number in data_set:
            a = number[0:self.no_of_digits]
            a = np.array(a)
            counts = np.bincount(a[0:self.no_of_digits])
            if (np.argmax(counts) == 0):
                number[no_of_digits] = -1
            else:
                number[no_of_digits] = 1
    
    def compute(self, number):
        return np.dot(self.weights, number[0:no_of_digits])

    def is_tag_right(self, number):
        result = self.compute(number)
        if result >= self.threshold:
            if number[no_of_digits] == 1:
                return True
        else:
            if number[no_of_digits] == -1:
                return True
        return False

    def configure(self, number):
        if number[no_of_digits] == 1:
            self.weights = self.weights + (number[0:no_of_digits] * self.learning_const)
        elif number[no_of_digits] == -1:
            self.weights = self.weights - (number[0:no_of_digits] * self.learning_const)

    def training(self):
        print("HOLD ON... We now train the neural network...\n\n************************************\n")
        counter = 0 #this variable will count the number of seccusefull recognizes by algorithm. 
        loop_counter = 0
        while (counter < self.data_amount - 1): # we train the perceptron untill until every number in data is well recognized.
            counter = 0
            loop_counter += 1
            #print(loop_counter)
            for number in self.data_set:
                if self.is_tag_right(number):
                    counter += 1
                else:
                    self.configure(number)
            print(f'succes on learning data set =  {(counter/self.data_amount) * 100}%')
        print("Training is done!")
        print("\n************************************\n")

    def algorithm_check(self):
        number = np.random.randint(0, 2, size = (1, self.no_of_digits + 1))
        number = number[0]
        print("this is our number to check - ", number)
        print("lets see what the perceptron say about it - ")
        a = number[0:self.no_of_digits]
        a = np.array(a)
        counts = np.bincount(a[0:self.no_of_digits])
        if (np.argmax(counts) == 0):
            number[no_of_digits] = -1
        else:
            number[no_of_digits] = 1
        result = self.compute(number)
        if result >= self.threshold:
            if number[no_of_digits] == 1:
                print("Hooray!!!!!!!!! perceptron do good! the number belongs to the 1's group")
            else:
                print("no luck today - perceptron was wrong about this one :((")
        else:
            if number[no_of_digits] == -1:
                print("Hooray!!!!!!!!! perceptron do good! the number belongs to the 0's group")
            else:
                print("no luck today - perceptron was wrong about this one :((")

    def wide_check(self):
        counter = 0
        print(f"\nnow let's test the perceptron on a different data set to see how succeful it is...\n the tested data is - \n{self.test_data_set}\n", )
        print("************************************\n")
        self.tagging(self.test_data_set)
        for number in self.test_data_set:
                if self.is_tag_right(number):
                    counter +=1
                else:
                    continue
        print(f'succes on a different data set =  {(counter/self.data_amount) * 100}%')


def main():
    p = Perceptron(data_amount = data_amount, learning_const = learning_const, with_bias = with_bias, no_of_digits = no_of_digits, threshold = threshold)
    p.tagging(p.data_set)
    print("Hello, I am a perceptron, A neural network created in order to classify a binary number with 21 digits into one of 2 groups:")
    print("1. one's group - if most of the digits are 1\n2. o's group - if most of the digits are 0\n")
    print("************************************\n")
    print(f"this is the data set:\n{p.data_set}\n\nnow we will train the algorithm on this data set.\n")
    p.training()
    print("this is weights after update - \n", p.weights)
    print("\n************************************\n")
    #p.algorithm_check()
    p.wide_check()



if __name__== "__main__":
  main()