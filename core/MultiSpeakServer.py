# -*- coding: utf-8 -*-
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import xml.etree.ElementTree as ET

from constants import *


# HTTPRequestHandler class
class testHTTPServer_RequestHandler(BaseHTTPRequestHandler):

    # GET
    def do_GET(self):
        # Send response status code
        self.send_response(200)

        # Send headers
        self.send_header('Content-type', 'text/html')
        self.end_headers()

        # Send message back to client
        message = "Hello world!"
        # Write content as utf-8 data
        self.wfile.write(bytes(message, "utf8"))
        return

    def do_POST(self):
        # set variable to information posted
        content_length = int(self.headers['Content-Length'])  # Gets the size of data
        post_data = self.rfile.read(content_length).decode('utf-8')  # Gets the data itself
        print("POST request,\nPath:", str(self.path))
        print("Headers:\n\n", str(self.headers))
        print("Body:\n", post_data)

        # save raw data as xml file for debugging
        with open(multispeak_raw_filename, 'w') as f:
            f.write(post_data)

        # parse data
        # TODO
        root = ET.fromstring(post_data)
        parent = root.find('parent')
        data = {}
        for element in parent:
            print(element, element.attrib, element.tag, element.text)
            data[element.tag] = element.text
        print(data)

        self.send_response(200)
        self.end_headers()

        # write dictionary to JSON file (overwrite existing JSON)
        app_json = json.dumps(data)
        with open(multispeak_filename, "w") as f:
            f.write(app_json)

        return


def run():
    # Choose port 8080, for port 80, which is normally used for a http server, you need root access
    PORT = 9000
    print('starting server on port', PORT)

    # Server settings
    server_address = ('0.0.0.0', PORT)
    httpd = HTTPServer(server_address, testHTTPServer_RequestHandler)
    print('running server on port', PORT)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print('Stopping httpd...\n')


if __name__ == "__main__":
    run()
