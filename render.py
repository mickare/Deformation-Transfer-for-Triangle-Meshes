

import pywavefront


cat_path = "models/lowpoly_cat/cat_reference.obj"
dog_path = "models/lowpoly_dog/dog_reference.obj"

cat = pywavefront.Wavefront(cat_path)
dog = pywavefront.Wavefront(dog_path)



print(cat)
