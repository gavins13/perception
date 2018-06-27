# evaluation model
 - At the moment, this is mainly done in execution.py. Instead, we will tell execution.py to call the evaluation model that will be defined in the Architecture Object. I.e. move this code from execution.py:
 
```python
 for i in range(len(results)):
          print("Saving results %s of %s" % (i, len(results)))
          path_orig = self.summary_folder + '/original/'
          path_mri = self.summary_folder + '/mri/'
          fig=plt.figure()
          plt.subplot(131)
          plt.imshow(input_data[i], cmap=plt.cm.gray)
          plt.subplot(132)
          plt.imshow(results[i], cmap=plt.cm.gray)
          plt.subplot(133)
          plt.imshow(ground_truths[i], cmap=plt.cm.gray)
          plt.savefig(path_mri + str(i) + ".png")
          plt.close(fig)
```

  to Architecture, along with the import matplotlib module!
