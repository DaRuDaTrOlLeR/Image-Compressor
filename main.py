# Image compression
#
# You'll need Python 3 and must install the 'numpy' package (just for its arrays).
#
# The code also uses the 'netpbm' package, which is included in this directory.
#
# You can also display a PNM image using the netpbm library as, for example:
#
#   python3 netpbm.py images/cortex.pnm
#
# NOTES:
#
#   - Use struct.pack( '>h', val ) to convert signed short int 'val' to a two-byte bytearray
#
#   - Use struct.pack( '>H', val ) to convert unsigned short int 'val' to a two-byte bytearray
#
#   - Use struct.unpack( '>H', twoBytes )[0] to convert a two-byte bytearray to an unsigned short int.  Note the [0].
#
#   - Use struct.unpack( '>' + 'H' * count, manyBytes ) to convert a bytearray of 2 * 'count' bytes to a tuple of 'count' unsigned short ints


import sys, os, math, time, struct, netpbm
from matplotlib import pyplot
import numpy as np


# Text at the beginning of the compressed file, to identify it

headerText = b'my compressed image - v1.0'



# Compress an image


def compress( inputFile, outputFile ):

  # Read the input file into a numpy array of 8-bit values
  #
  # The img.shape is a 3-type with rows,columns,channels, where
  # channels is the number of component in each pixel.  The img.dtype
  # is 'uint8', meaning that each component is an 8-bit unsigned
  # integer.

  img = netpbm.imread( inputFile ).astype('uint8')

  # Note that single-channel images will have a 'shape' with only two
  # components: the y dimensions and the x dimension.  So you will
  # have to detect whether the 'shape' has two or three components and
  # set the number of channels accordingly.  Furthermore,
  # single-channel images must be indexed as img[y,x] instead of
  # img[y,x,k].  You'll need two pieces of similar code: one piece for
  # the single-channel case and one piece for the multi-channel case.

  startTime = time.time()

  # Compress the image
  outputBytes = bytearray()

  # ---------------- [YOUR CODE HERE] ----------------
  #
  # REPLACE THE CODE BELOW WITH YOUR OWN CODE TO FILL THE 'outputBytes' ARRAY.

  # Check if the image has a single channel or multiple channels
  if len(img.shape) == 2:
    img = img.reshape(img.shape[0],img.shape[1],1)

  # Initialize the LZW dictionary with all possible strings of length 1
  LZW_dict = {}
  for i in range(-255, 256):
    LZW_dict[bytes(str(i), encoding='utf-8')] = struct.pack('>H', i + 255)

  dicSize = len(LZW_dict)
  maxSize = 65536

  # Compress the image
  s = b""
  for y in range(img.shape[0]):
    for x in range(img.shape[1]):
      for c in range(img.shape[2]):
        # Predictive encoding
        if y != 0:
          pred = img[y-1, x, c]
        else:
          pred = 0

        # Add the difference to the difference array
        dif = img[y, x, c] - int(pred)

        # Encode the difference array using LZW compression
        code = bytes(str(dif), encoding='utf-8')
        tmp_s = s + b"," + code
        if tmp_s in LZW_dict:
            s = tmp_s
        else:
            # Check if empty
            if s != b"":
              # add LZW code to the output
              outputBytes += LZW_dict[s]
            # Add the new code to the dictionary if it's not full
            if dicSize < maxSize:
                LZW_dict[tmp_s] = struct.pack('>H', dicSize)
                dicSize += 1

            s = code

  # Add the final LZW code to the output
  outputBytes += LZW_dict[s]

  endTime = time.time()

  # Output the bytes
  #
  # Include the 'headerText' to identify the type of file.  Include
  # the rows, columns, channels so that the image shape can be
  # reconstructed.

  outputFile.write( headerText + b'\n' )
  outputFile.write( bytes( '%d %d %d\n' % (img.shape[0], img.shape[1], img.shape[2]), encoding='utf8' ) )
  outputFile.write( outputBytes )

  # Print information about the compression
  
  inSize  = img.shape[0] * img.shape[1] * img.shape[2]
  outSize = len(outputBytes)

  sys.stderr.write( 'Input size:         %d bytes\n' % inSize )
  sys.stderr.write( 'Output size:        %d bytes\n' % outSize )
  sys.stderr.write( 'Compression factor: %.2f\n' % (inSize/float(outSize)) )
  sys.stderr.write( 'Compression time:   %.2f seconds\n' % (endTime - startTime) )
  


# Uncompress an image

def uncompress( inputFile, outputFile ):

  # Check that it's a known file

  if inputFile.readline() != headerText + b'\n':
    sys.stderr.write( "Input is not in the '%s' format.\n" % headerText )
    sys.exit(1)
    
  # Read the rows, columns, and channels.  

  rows, columns, numChannels = [ int(x) for x in inputFile.readline().split() ]

  # Read the raw bytes.

  inputBytes = bytearray(inputFile.read())

  startTime = time.time()

  # ---------------- [YOUR CODE HERE] ----------------
  #
  # REPLACE THIS WITH YOUR OWN CODE TO CONVERT THE 'inputBytes' ARRAY INTO AN IMAGE IN 'img'.

  # Convert the inputBytes array into an image in 'img'
  #img = np.empty([rows, columns, numChannels], dtype=np.uint8)

  # Initialize output array
  data = np.empty(rows * columns * numChannels)

  # Initialize the LZW dictionary with all possible strings of length 1
  LZW_dict = {}
  for i in range(256):
    LZW_dict[i] = np.array([i], dtype=np.uint8)

  # First code
  code = struct.unpack('>B', inputBytes[5:6])[0]
  s = LZW_dict[code]

  # Initialize output list
  data = [s]

  # Loop through inputBytes
  i = 6
  pos = 0
  while i < len(inputBytes):
    # Extract next code
    code = struct.unpack('>B', inputBytes[i:i+1])[0]
    i += 1

    # Check if in dictionary
    if code in LZW_dict:
      entry = LZW_dict[code]
    else:
      entry = np.concatenate([s, np.array([s[0]], dtype=np.uint8)])

    # Add new entry to dictionary
    LZW_dict[len(LZW_dict)] = np.concatenate([s, np.array([entry[0]], dtype=np.uint8)])

    # Output entry
    data[pos:pos + len(entry)] = entry
    pos += len(entry)

    # Set s to current entry
    s = entry

  # Flatten list of img
  data = np.concatenate(data)
  data = data[:rows * columns * numChannels]

  # Turn into img array
  img = data.reshape(rows, columns, numChannels)


  # ---------------- [END OF YOUR CODE] ----------------

  endTime = time.time()
  sys.stderr.write('Uncompression time %.2f seconds\n' % (endTime - startTime))

  netpbm.imsave(outputFile, img)

# The command line is 
#
#   main.py {flag} {input image filename} {output image filename}
#
# where {flag} is one of 'c' or 'u' for compress or uncompress and
# either filename can be '-' for standard input or standard output.


if len(sys.argv) < 4:
  sys.stderr.write( 'Usage: main.py c|u {input image filename} {output image filename}\n' )
  sys.exit(1)

# Get input file
 
if sys.argv[2] == '-':
  inputFile = sys.stdin
else:
  try:
    inputFile = open( sys.argv[2], 'rb' )
  except:
    sys.stderr.write( "Could not open input file '%s'.\n" % sys.argv[2] )
    sys.exit(1)

# Get output file

if sys.argv[3] == '-':
  outputFile = sys.stdout
else:
  try:
    outputFile = open( sys.argv[3], 'wb' )
  except:
    sys.stderr.write( "Could not open output file '%s'.\n" % sys.argv[3] )
    sys.exit(1)

# Run the algorithm

if sys.argv[1] == 'c':
  compress( inputFile, outputFile )
elif sys.argv[1] == 'u':
  uncompress( inputFile, outputFile )
else:
  sys.stderr.write( 'Usage: main.py c|u {input image filename} {output image filename}\n' )
  sys.exit(1)
