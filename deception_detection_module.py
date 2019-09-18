# ADDITIONAL FUNCTIONS WRITTEN IN THE MEANTIME
# POTENTIALLY TO BE USED IN LATER STAGES

def vect_and_concat(vecMod, input = [], ax = None):
  '''
  Function takes tree arguments:
      vectMod:
          requires a function that returns a vectorized version of the
          elements in input
      input:
          list of elements to be vectorized and concatenated
      ax:
          axis alon which the vector will be concatenated, for further
          information check documentation of np.concatenate()
          
  Function returns:
      output_vector:
          NumPy array of the elements of the input vectorized and concatenated
          along the chozen axix
  '''
  
  output_vector = np.array([])
  for element in input:
    try:
      output_vector = np.concatenate(([output_vector], [vecMod[element]]), ax)
    except:
      pass
  return output_vector


def filter(input_vector = [], filter_size = 1):
  '''
  Function takes two arguments:
      input_vector:
          one-dimentional, iterable variable of numbers
      filter_size:
          size of filter applied on elements of the input vector
          
  Function returns:
      output_vector:
          filtered list of size input_vector/filter_size rounded up
  '''
  
  output_vector = list()
  
  # Number of iterations neccessary to shorten the input to 'size'
  iter_number = len(input_vector) - filter_size + 1
  
  for i in range(iter_number):
    output_vector.append(sum(input_vector[i:i+filter_size])/filter_size)
    
  return output_vector


def short_vect(input_vector, size, filter_size = 1):
  '''
  Function takes three arguments:
      input_vector:
          one-dimentional, iterable variable of numbers to be resized
      size:
          desirable size of the output vector
      filter_size:
          size of filter applied on elements of the input vector
        
  Function returns:
      output_vector:
          filtered NumPy array, a list of numbers
        
          if the output_size would be less than desirable size, because of the
          filtering, the output is appended with zeros for now, will  be changed
          if needed
  '''
  from numpy import array
  
  output_vector = input_vector
  
  while len(output_vector) > size:
    output_vector = filter(output_vector, filter_size)
  
  while len(output_vector) < size:
    output_vector.append(0)
    
  return output_vector


def remove_elements(input = [], elements_set = []):
  '''
  Function takes two arguments:
    text:
        iterable variable from which the elements will be removed
    elements_set:
        iterable variable of elements to remove
    
  Function returns:
      NumPy array made of input, without the elements from elements_set
  '''
  from numpy import array
  
  return array([item for item in input if item not in set(elements_set)])
