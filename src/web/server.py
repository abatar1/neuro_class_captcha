import socket
import model_prediction as mp

server_sockset = socket.socket()
host = socket.gethostname()
port = 9000
server_sockset.bind((host,port))
filename = ""
server_sockset.listen(10)
print "Waiting for a connection..."

client_socket, addr = server_sockset.accept()
print("Got a connection from %s." % str(addr))
while True:
    size = client_socket.recv(16)
    if not size:
        break
    size = int(size, 2)
    filename = client_socket.recv(size)
    filesize = client_socket.recv(32)
    filesize = int(filesize, 2)
    file_to_write = open(filename, 'wb')
    chunksize = 4096
    while filesize > 0:
        if filesize < chunksize:
            chunksize = filesize
        data = client_socket.recv(chunksize)
        file_to_write.write(data)
        filesize -= chunksize

    file_to_write.close()
    print 'File received successfully. Getting started...'

    predictor = mp.ModelPrediction()
    prediction = predictor.predict(filename)
    client_socket.send(prediction)

    print 'Captcha successfully predicted.'

server_sockset.close()