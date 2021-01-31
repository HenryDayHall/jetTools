def fixBadZipfile(zipFile):  
 f = open(zipFile, 'r+b')  
 content = f.read()  
 pos = content.rfind(b'\x50\x4b\x05\x06') # reverse find: this string of bytes is the end of the zip's central directory.
 if pos>0:
     f.seek(pos+20) # +20: see secion V.I in 'ZIP format' link above.
     f.truncate()
     f.write(b'\x00\x00') # Zip file comment length: 0 byte length; tell zip applications to stop reading.
     f.seek(0)
     f.close()  
 else:  
     # raise error, file is truncated  
     print("probelm")
