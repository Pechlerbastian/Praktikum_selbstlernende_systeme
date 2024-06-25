/*
This document uses the express.js framework to set up a webserver. From the current running js process the input is
read and send as message for the companion using a websocket. Moreover, it is specified to display the index.html
on the webserver.
@author: Marcel Achner
 */


const express = require('express');
const app = express();
const http = require('http');
const server = http.createServer(app);
const { Server } = require("socket.io");
const io = new Server(server);
const readline = require('readline');

// creating an interface for communicating via the stdin and stdout of the process
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  terminal: false
});

rl.on('line', function (line) {
  const data = JSON.parse(line);
  io.emit('avatar_json', data);
});

app.get('/', (req, res) => {
  res.sendFile(__dirname + '/index.html');
});

server.listen(3000, () => {
  console.log('listening on *:3000');
});