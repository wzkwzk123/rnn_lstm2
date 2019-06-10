(1) the lstm model is to understand better, the lstm_matrix is to make the code looks simple. no big difference

(2) problem: the loss is ver hard to decrease after it reaches about 5, I haven't found a good solution

(3) !!! 
at first the training result is very bad, because there is a mistake in the code. （the result is showed in this file :lstm_main_ifoc(ergebnis with mistake) ）

I didn't write the following code, and I reshape the data directly. the result is : there is no syntax error, but the data is already destroyed.
out = out.permute(1,0,2) ##  because of this line the result is much better

We must not only pay attention to the dimensions to meet the syntax requirements, but also to meet the actual meaning.

a = torch.Tensor ([[1,4],[2,5],[3,6]])
print(a)
tensor([[1., 4.],
        [2., 5.],
        [3., 6.]])

（4） 和 tutor确认
一句话24个单词共用同一个h_pre 和  c_pre
b = a.reshape(6,-1)  ## directly reshape, the meaning the data is destroyed.
print(b)
tensor([[1., 4., 2.],
        [5., 3., 6.]])

c = a.permute(1,0)
print(c)
tensor([[1., 2., 3.],
        [4., 5., 6.]])

after this we can reshape to one column.
 in the code the dim is bigger than 2, so we use permute instead of .t()