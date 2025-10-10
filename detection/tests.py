import pika,sys


connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

channel.queue_declare(queue='hello')

message = ' '.join(sys.argv[1:]) or "Hello world!"
channel.basic_publish(exchange='',
                      routing_key='hello',
                      body=message)

print(" sent Hello World!")

connection.close()