# ---------------- [YOUR CODE HERE] ----------------
#
# REPLACE THE CODE BELOW WITH YOUR OWN CODE TO FILL THE 'outputBytes' ARRAY.

# Initialize the LZW dictionary with all possible strings of length 1
LZW_dict = {}
for i in range(-255, 256):
    LZW_dict[str(i)] = i + 255

dicSize = len(LZW_dict)
maxSize = 65536

# Check if the image has a single channel or multiple channels
if len(img.shape) == 2:
    img = img.reshape(img.shape[0], img.shape[1], 1)

# Compress the image
s = ""
for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        for c in range(img.shape[2]):
            # Predictive encoding
            if y != 0:
                pred = img[y - 1, x, c]
            else:
                pred = 0

            # Add the difference to the difference array
            dif = img[y, x, c] - int(pred)

            # Encode the difference array using LZW compression
            code = str(dif)
            tmp_s = s + "," + code
            if tmp_s in LZW_dict:
                s = tmp_s
            else:
                # Check if empty
                if s != "":
                    # add LZW code to the output
                    outputBytes += struct.pack('>H', LZW_dict[s])
                # Add the new code to the dictionary if it's not full
                if dicSize < maxSize:
                    LZW_dict[tmp_s] = dicSize
                    dicSize += 1

                s = code

# Add the final LZW code to the output
outputBytes += struct.pack('>H', LZW_dict[s])

# ---------------- [END OF YOUR CODE] ----------------
