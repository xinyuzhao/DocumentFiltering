import docclass

if __name__ == '__main__':
   cl=docclass.naiveBayes(docclass.getWords) 
   docclass.sampleTrain(cl)
   
   tmp = cl.classify('quick rabbit', default='unknown')
   print(tmp)

   tmp = cl.classify('quick money', default='unknown')
   print(tmp)

   cl.setThreshold('bad', 3.0)
   tmp = cl.classify('quick money', default='unknown')
   print(tmp)

   for i in range(10):
       docclass.sampleTrain(cl)
   
   tmp = cl.classify('quick money', default='unknown')
   print(tmp)
