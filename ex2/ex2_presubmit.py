import os
import numpy as np
import current.sol2 as sol2

def presubmit():
  print ('ex2 presubmission script')
  disclaimer="""
  Disclaimer
  ----------
  The purpose of this script is to make sure that your code is compliant
  with the exercise API and some of the requirements
  The script does not test the quality of your results.
  Don't assume that passing this script will guarantee that you will get
  a high grade in the exercise
  """
  print (disclaimer)
  
  if not os.path.exists('current/README'):
    print ('No readme!')
    return False
  with open ('current/README') as f:
    lines = f.readlines()
  print ('login: ', lines[0])
  print ('submitted files:\n' + '\n'.join(lines[1:]))
  
  for q in [1,2,3]:
    if not os.path.exists('current/answer_q%d.txt'%q):
      print ('No answer_q%d.txt!'%q)
      return False
    print ('answer to q%d:'%q)
    with open('current/answer_q%d.txt'%q) as f:
      print (f.read())
  
  print ('section 1.1')  
  print ('DFT and IDFT')
  filename = 'external/monkey.jpg'
  try:
    im = sol2.read_image(filename, 1)
    dft_1d = sol2.DFT(im[:,0])
    if not np.all(dft_1d.shape==im[:,0].shape):
      print ('Failed!')
      return False
    idft_1d = sol2.IDFT(dft_1d)
    if not np.all(idft_1d.shape==dft_1d.shape):
      print ('Failed!')
      return False
    if idft_1d.dtype != np.complex:
      print ('IDFT should return complex values!')
      print ('Failed!')
      return False
  except:
    print ('Failed!')
    return False
  
  print ('section 1.2')  
  print ('2D DFT and IDFT')
  try:
    dft = sol2.DFT2(im)
    if not np.all(dft.shape==im.shape):
      print ('Failed!')
      return False
    idft = sol2.IDFT2(dft)
    if not np.all(idft.shape==dft.shape):
      print ('Failed!')
      return False
    if idft.dtype != np.complex:
      print ('IDFT should return complex values!')
      print ('Failed!')
      return False
  except:
    print ('Failed!')
    return False
  
  print ('section 2.1')
  print ('derivative using convolution')
  try:
    magnitude = sol2.conv_der(im)
    if not np.all(magnitude.shape==im.shape):
      print ('derivative magnitude shape should be :', im.shape, 'but is:' , magnitude.shape)
      print ('Failed!')
      return False
  except:
    print ('Failed!')
    return False
  
  print ('Section 2.2')
  print ('derivative using convolution')
  try:
    magnitude = sol2.fourier_der(im)
    if not np.all(magnitude.shape==im.shape):
      print ('derivative magnitude shape should be :', im.shape, 'but is:' , magnitude.shape)
      print ('Failed!')
      return False
  except:
    print ('Failed!')
    return False
  
  print ('Section 3.1')
  try:
    print ('blur spatial')
    blur_im = sol2.blur_spatial (im, 5)
    if not np.all(blur_im.shape==im.shape):
      print ('blured image''s shape should be :', im.shape, 'but is:' , blur_im.shape)
      print ('Failed!')
      return False
  except:
    print ('Failed!')
    return False
  
  print ('Section 3.1')
  try:
    print ('blur fourier')
    blur_im = sol2.blur_fourier (im, 5)
    if not np.all(blur_im.shape==im.shape):
      print ('blured image''s shape should be :', im.shape, 'but is:' , blur_im.shape)
      print ('Failed!')
      return False
  except:
    print ('Failed!')
    return False

  print ('all tests Passed.');
  print ('- Pre-submission script done.');
  
  print ("""
  Please go over the output and verify that there are no failures/warnings.
  Remember that this script tested only some basic technical aspects of your implementation
  It is your responsibility to make sure your results are actually correct and not only
  technically valid.""")
  return True





