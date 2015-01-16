import socket
from threading import Thread
from sys import exit
import os
from datetime import datetime
import multiprocessing


global port = 6667


class Bot():

	def __init__(self, network, channel, nick, password):

		self.s = socket.socket()
		self.s.settimeout(300)

		self.password = password
		self.nick = nick
		self.channel = channel
		self.network = network

		self.connected = False


	def connect(self):

		self.s.connect((network, port))

		self.s.send('PASS %s\r\n' % self.password )
		self.s.send('USER %s 0 * : owner\r\n' % self.nick )
		self.s.send('NICK %s\r\n' % self.nick )
		self.s.send('JOIN %s\r\n' % self.channel)
		
		self.s.send('PRIVMSG %s : Hi! I\'m Potato Bot!\r\n' % self.channel)



	def recv_data(self):
		
		data = self.s.recv(1024)

		while True: # while connected, check with self.connect()
			data = self.s.recv(1024)

			if len(data) == 0:
				break
				# instead of break, make it reconnect to server

			self.parse_data(data)
		self.s.close()
		exit(0)

	def pong(self, data):
		pong_server = data.split()[1]
		print "Pong sent at :" + str(datetime.now()) 
		self.s.send('PONG ' + pong_server + "\r\n")
		return

	def parse_data(self, data):

		data = data

		if 'PING' in data:
			self.pong(data)
			return


		# print data

		if 'PRIVMSG' in data:
			try:
				split = data.split(" ",3)

				user = split[0].split(":")[1].split("!")[0]
				command = split[1].upper()
				channel = split[2].replace("#","")
				msg = split[3]
				msg = msg.replace(msg[0], "")

				data = "[+] %s: %s" % (user, msg)

				print data

				if msg.lower() == "!leave\r\n":
					print "ending..."
					self.s.close()
					exit(0)

				# addcomS = "!addcom"

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
					except:
						pass
			except:
				pass
		else:
			print data

	def get_comms(self):
		with open("commands.txt", "r") as f:
			all_comms = [a.split(":", 1)[0] for a in f.readlines()]
			return all_comms

	def debug_environment(self):
		pass
		# make @command admin only
		# if admin types @test_environment, switch interpretation by bot to raw (no parsing)
		# allow admin to test raw commands in twitch chat


	def display_comms(self):
		all_comms = self.get_comms()
		all_comms = str(all_comms).strip('[]').replace("'","") # turns from a list to text for user

		self.s.send("PRIVMSG "+channel+" :Commands: "+all_comms+"\r\n")


	# make a new "!addcomO" that overwrites existing command(s)

	def check_command(self, name, text):
		name = name
		text = text

		print "checking command..."
		print "name:%s text:%s"%(name, text)

		all_comms = self.get_comms()


		if name in all_comms:
			print "command already exists..."
			self.s.send("PRIVMSG %s :%s already exists.\r\n" % channel,)
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
		name = name.strip("\r\n")
		print "starting usecommand: %s" % name

		f = open("commands.txt", "r")
		comm_list = f.readlines()

		for comms in comm_list:
			comms = comms.split(":", 2)

			Command = comms[0] # actual command
			Text = comms[1].strip('\n') # what command outputs


			if str(name).lower() == "!comms":
				self.display_comms()
				f.close()
				return


			elif str(name).lower() == str(Command).lower():
				print Text
				self.s.send("PRIVMSG #potatozero :"+Text+"\r\n")	
				f.close()
				return


		f.close()

	def newcommand(self, command_name, command_text):
		command_name = command_name
		command_text = command_text
		whattowrite = command_name+":"+command_text+"\n"

		all_comms = self.get_comms()


		if command_name in all_comms:
			print "exists already"
			self.s.send("PRIVMSG %s :%s already exists."% (channel, str(command_name)))
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


		self.start_connection()


network = 'irc.twitch.tv'


channel = '#potatozero'
oauth = 'oauth:zayj9ocfmxvvqzx34n8yj07m8pvi5i'
nick = 'tobotatop'

if __name__ == '__main__':
	bot = Bot(network, channel, nick, oauth)
	bot.start()

# TODO
# CONNECT THE IRC BOT TO THE ADMIN'S ACCOUNT USING HWID OR IP CHECK (BAD IDEA, FIND ALTERNATIVE)
# MAKES SURE YOU CAN'T USE THIS BOT UNLESS YOU ARE ON THE USER'S COMPUTER
# ONCE A COMPUTER (IP ADDRESS) HAS BEEN "ESTABLISHED", IF ANOTHER IP ADDRESS REQUESTS ACCESS TO BOT, SEND EMAIL LIKE STEAM

# USE PYTHON LOGGING TO KEEP TRACK OF WHAT COMMANDS HAVE BEEN USED