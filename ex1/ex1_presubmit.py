import os
import current.sol1 as sol1

def presubmit():
  print ('ex1 presubmission script')
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
  
  if not os.path.exists('current/answer_q1.txt'):
    print ('No answer_q1.txt!')
    return False
  print ('answer to q1:')
  print (open('current/answer_q1.txt').read())
  
  filename = 'external/monkey.jpg'
  print ('section 3.1')
  
  print ('Reading images')
  try:
    im_rgb = sol1.read_image(filename, 2)
    im_gray = sol1.read_image(filename, 1)
  except:
    print ('Failed!')
    return False
  
  print ('section 3.3')
  print ('Transforming rgb->yiq->rgb')
  try:
    imYIQ = sol1.rgb2yiq(im_rgb)
    sol1.yiq2rgb(imYIQ)
  except:
    print ('Failed!')
    return False
  
  print ('Section 3.4')
  try:
    print ('- Histogram equalization...')
    im_orig = sol1.read_image('external/Low Contrast.jpg', 2)
    im_eq, hist_orig, hist_eq = sol1.histogram_equalize(im_orig)
    if hist_orig.size is not 256 or hist_eq.size is not 256:
      print ('incorrect number of bins in histogram') 
      print ('Failed!')
      return False
  except:
    print ('Failed!')
    return False
  
  print ('Section 3.5')
  try:
    print ('- Image quantization...')
    im_orig = sol1.read_image('external/jerusalem.jpg', 1);
    im_quant, err = sol1.quantize(im_orig, 6, 3);
    if len(err) is not 3:
      print ('incorrect number of elements in err') 
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





