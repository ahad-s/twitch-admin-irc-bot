import socket
from threading import Thread
from sys import exit
import sys
import os
from datetime import datetime
import multiprocessing

import tensorflow as tf


port = 6667

network = 'irc.twitch.tv'

nick = 'tobotatop'



class Bot(object):

	def __init__(self, network, channel, nick, password, scraper=False):

		self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.s.settimeout(300)

		self.password = password
		self.nick = nick
		self.channel = channel
		self.network = network
		self.scraper = scraper

		# print password, nick, channel, network

		self.connected = False

		self.log_file = open(channel[1:] + ".txt", "a+") if self.scraper else None


	def connect(self):

		self.s.connect((network, port))

		self.s.send('PASS %s\r\n' % self.password )
		self.s.send('USER %s 0 * : owner\r\n' % self.nick )
		self.s.send('NICK %s\r\n' % self.nick )
		self.s.send('JOIN %s\r\n' % self.channel)
		if not self.scraper:
			self.s.send('PRIVMSG %s : Hi! I\'m Potato Bot! Type !help if u a noob.\r\n' % self.channel)

	def recv_data(self):
		
		# data = self.s.recv(1024)

		while True: # while connected, check with self.connect()
			data = self.s.recv(1024)

			print data

			if len(data) == 0:
				break
				# instead of break, make it reconnect to server

			if 'PING' in data:
				self.pong(data)
				continue

			if self.scraper:
				self.scrape_data(data)
			else:
				self.parse_data(data)

		self.s.close()
		print "fail"
		exit(0)

	def pong(self, data):
		pong_server = data.split()[1]
		print "Pong sent at :" + str(datetime.now()) 
		self.s.send('PONG ' + pong_server + "\r\n")
		return


	def parse_raw_privmsg(self, data):
		split = data.split(" ",3)

		user = split[0].split(":")[1].split("!")[0]
		command = split[1].upper()
		channel = split[2].replace("#","")
		msg = split[3][1:]
		# msg = msg[1:]

		return user, command, channel, msg


	def scrape_data(self, data):
		if 'PRIVMSG' in data:
			try:
				usr, cmd, chann, msg = self.parse_raw_privmsg(data)
				self.log_file.write(msg)
			except Exception as e:
				print e
		else:
			print data

	def parse_data(self, data):

		if 'PRIVMSG' in data:
			try:
				user, command, channel, msg = self.parse_raw_privmsg(data)

				data = "[+] %s: %s" % (user, msg)

				if msg.lower() == "!leave\r\n":
					self.display("PEACE OUT Y'ALL")
					self.s.close()
					exit(0)

				command = msg.split(" ", 2)[0]

				if '!addcom' == command:
					command_name = msg.split(" ", 2)[1]
					command_text = (msg.split(" ", 2)[2]).strip("\r\n")
					print "name:%s text:%s"%(command_name, command_text)
					self.check_command(command_name, command_text)
					return

				if '!delcom' == command:
					command_name = msg.split(" ", 1)[1]
					print "del command name: %s" % command_name
					self.del_command(command_name)
					return

				if str(command).startswith('!'):
					try:
						self.use_command(command)
						return
					except Exception as e:
						print e
			except Exception as e:
				print e
		else:
			print data
		# print 'NAMES %s\r\n' % self.channel
		# self.s.send('NAMES %s\r\n' % self.channel)
		# print self.s.recv(1024)

	def get_comms(self):
		with open("commands.txt", "r") as f:
			all_comms = [a.split(":", 1)[0] for a in f.readlines()]
			return all_comms

	def debug_environment(self):
		pass
		# make @command admin only
		# if admin types @test_environment, switch interpretation by bot to raw (no parsing)
		# allow admin to test raw commands in twitch chat

	def display(self, text):
		self.s.send("PRIVMSG " + channel + " :" + text + "\r\n")

	def display_comms(self):
		all_comms = self.get_comms()
		all_comms = str(all_comms).strip('[]').replace("'","") # turns from a list to text for user
		self.display("Commands: " + all_comms)

	def display_help(self):
		text = "Type \"!comms\" to check \
			out all available commands! Type \"![command]\" to use the command!\
			Type \"!addcomm [command_name] [command_text]\" to add a new command!\
			Type \"!delcom [command_name]\" to delete commands.\
			Currently PotatoBot only supports text commands :("
		self.display(text)
	# make a new "!addcomO" that overwrites existing command(s)

	def check_command(self, name, text):
		name = name
		text = text

		print "checking command..."
		print "name:%s text:%s"%(name, text)

		all_comms = self.get_comms()

		if name in all_comms:
			print "command already exists..."
			self.s.send("PRIVMSG " + channel + " :" + name +" already exists.\r\n")
			return
		else:
			print "making new command..."
			self.newcommand(name, text)
			return

	def del_command(self, name):
		
		name = name.strip("\r\n")


		fr = open("commands.txt", "r")
		all_comms = [b.split(":", 1) for b in fr.readlines()] # returns 2d list with command:text
		# fr.seek(0) # moves it back to beginning of file
		# temp_data = fr.read()
		fr.close()


		for i in range(len(all_comms)):
			print all_comms[i]

			# print all_comms[i][0]
			if str(name) == str(all_comms[i][0]):
				print "removing... %s\n" % all_comms[i]
				all_comms.remove(all_comms[i])
				break


		num = 2
		temp = None

		t = open("commands.txt","w")

		for a in all_comms:
			for b in a:

				if temp != None:
					print "writing... " + temp + ":" + b
					t.write("%s:%s"%(temp, b))
				if (num % 2) == 0:
					temp = b
				else:
					temp = None

				num+=1

		t.close()

	def use_command(self, name):
		name = name.strip("\r\n")[1:] # removes ! at beginning and \n at end
		print "starting usecommand: %s" % name

		f = open("commands.txt", "r")
		comm_list = f.readlines()

		found = False

		for comms in comm_list:
			comms = comms.split(":", 1)

			Command = comms[0] # actual command
			Text = comms[1].rstrip('\n') # output of command
			print "command is [%s] -- text is [%s]" % (Command, Text)

			print "name [%s] -- command [%s]" %(str(name).lower(), str(Command).lower())
			try:
				if str(name).lower() == "comms":
					self.display_comms()
					found = True
				elif str(name).lower() == "help":
					print "displaying help..."
					self.display_help()
					found = True
				elif str(name).lower() == str(Command).lower():
					print Text
					self.s.send("PRIVMSG #potatozero :"+Text+"\r\n")	
					found = True

				if found:
					f.close()
					return
			except Exception as e:
				print e



		f.close()

	def newcommand(self, command_name, command_text):
		command_name = command_name
		command_text = command_text
		whattowrite = command_name+":"+command_text+"\n"

		all_comms = self.get_comms()

		if command_name == "comms" or command_name == "help":
			self.s.send("PRIVMSG " + channel + " :Stop it.")
			return


		if command_name in all_comms:
			print "exists already"
			self.s.send("PRIVMSG " + command + " :" + command_name + " already exists.")
			return
			

		f = open("commands.txt", "a")
		f.write(whattowrite)
		f.close()


	def start_connection(self):

		data_thread = Thread(target = self.recv_data())

		data_thread.setDaemon(True)

		data_thread.start()

	def start(self):
		# while not self.connected: -- EDIT THIS TO ENSURE THAT IT'S ALWAYS CONNECTED
		self.connect()
		print "connected..."
		self.start_connection()


class Scraper(object):

	# channel = channel to scrape 	
 	def __init__(self, channel):
 		self.b = Bot(network, channel, nick, oauth, scraper=True)

 	def start(self):
 		self.b.start()



if __name__ == '__main__':
	print sys.argv
	if len(sys.argv) == 1:
		channel = '#potatozero'
	else:
		channel = '#' + str(sys.argv[1])

	# bot = Bot(network, channel, nick, oauth)
	# bot.start()

	scpr = Scraper(channel)
	scpr.start()
