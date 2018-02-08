import sys
import script
epochs=sys.argv[1]
script.preprocess()
script.run(int(epochs))
