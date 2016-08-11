from __future__ import absolute_import
from __future__ import unicode_literals
from autobahn.twisted.websocket import WebSocketClientProtocol, WebSocketClientFactory
from twisted.internet import reactor
import threading
import datetime
import sys
import json
import traceback
import time
import numpy as np
import robot_controller

iconnector = None
factory = None


def get_open_port_cmd(portname):
    return json.dumps({"cmd": "put", "api": "openSerialPort", "port": portname}, ensure_ascii=False).encode('utf-8')


def get_w_motor_angle_cmd(partid, angle, speed):
    return json.dumps(
        {"cmd": "put", "api": "motorWrite", "partid": partid, "servoid": 1, "angle": angle, "speed": speed},
        ensure_ascii=False).encode('utf-8')


CMD_GET_ALL_PORT = json.dumps({"cmd": "get", "property": "serialports", "payload": [], "api": "getSerialPorts"},
                              ensure_ascii=False).encode('utf-8')
CMD_OPEN_PORT = get_open_port_cmd
CMD_POWER_ON = json.dumps({"cmd": "put", "api": "powerOn"}, ensure_ascii=False).encode('utf-8')
CMD_W_MOTOR_ANGLE = get_w_motor_angle_cmd

th_arm_main_loop = None
stop_event = threading.Event()

portstate = 'close'
portname = ''


class MyClientProtocol(WebSocketClientProtocol):
    def connectionLost(self, reason):
        # callback from WebSocketClientProtocol
        print ("Connection lost: %s" % reason.value)
        self._connectionLost(reason)

    def onConnect(self, response):
        print("Server connected: {0}".format(response.peer))

    def onOpen(self):
        print("WebSocket connection open.")

        ''' def hello():
            self.sendMessage(u"Hello, world!".encode('utf8'))
            self.sendMessage(b"\x00\x01\x03\x04", isBinary=True)
            self.factory.reactor.callLater(1, hello)
        # start sending messages every second ..
        hello()
        '''
        self.factory._proto = self

    def onMessage(self, payload, isBinary):

        # print('isBinaly type: %s' % type(isBinary))
        # print('isBinaly: %s' % isBinary)
        if isBinary:
            print("Binary message received: {0} bytes".format(len(payload)))
            self.sendMessage(payload, isBinary)
        else:
            data = json.loads(payload.decode('utf8'))
            if data['api'] == 'getSerialPorts':
                print("seriap ports: {}".format(data['payload']))
            if data['api'] == 'openSerialPort':
                print ('open port: {}'.format(data['port']))
                global portstate
                portstate = 'open'
                portname = data['port']
                # print('payload type: %s' % type(payload)) #bytes
                print("Text message received: {0}".format(payload.decode('utf8')))

    def onClose(self, wasClean, code, reason):
        print("WebSocket connection closed: {0}".format(reason))


def my_cleanup(name):
    print('my_cleanup(%s)' % name)
    if (iconnector != None):
        iconnector.disconnect()


def hoge(n, t):
    print " === start sub thread (method) === "
    print "[%s] sub thread (method) : " % threading.currentThread().getName() + str(datetime.datetime.today())
    while True:
        cmd = raw_input('>>>  ').split()
        try:
            if cmd[0] == 'ports':
                print(cmd[0])
                reactor.callFromThread(factory._proto.sendMessage, CMD_GET_ALL_PORT)
            elif cmd[0] == 'open':
                portName = cmd[1]
                print('open')
                reactor.callFromThread(factory._proto.sendMessage, CMD_OPEN_PORT(portName))
            elif cmd[0] == 'power':
                ctrl = cmd[1]
                reactor.callFromThread(factory._proto.sendMessage, CMD_POWER_ON)
            elif cmd[0] == 'wm':
                partid = cmd[1]
                angle = cmd[2]
                speed = cmd[3]
                reactor.callFromThread(factory._proto.sendMessage,
                                       CMD_W_MOTOR_ANGLE(int(partid), int(angle), int(speed)))
            elif cmd[0] == 'help':
                print('Example usage')
                print('  ports')
                print('  open [port name]')
                print('  power [on|off]')
                print('  wm [part id] [angle] [speed]')
                print('  q')
            elif cmd[0] == 'leap':
                print('start!')
            elif cmd[0] == 'q':
                print('finish')
                reactor.callFromThread(reactor.stop)
                stop_event.set()
                break
            else:
                print('unknown command')
        except:
            print("error message: {}".format(sys.exc_info()[0]))
            print(traceback.format_exc())

    print " === end sub thread (method) === "


class PlumManager:
    def __init__(self):
        self.t = None
        self._init_angs_rad = np.matrix(np.deg2rad([25, 15, -35, 60])).T

        self._robot = robot_controller.Robot(self._init_angs_rad)
        self._leap_mgr = robot_controller.LeapManager(np.matrix([0, -100, 0]).T)
        self._robot_controller = robot_controller.RobotController(self._init_angs_rad)
        self._robot_visualizer = robot_controller.Visualizer()

        self._target_pos = self._robot.joint_pos[-1]

    def init_motor(self):
        self.send_angles(self._init_angs_rad)
        self.t = None

    def update(self):
        if self._leap_mgr.update():
            self._target_pos = self._leap_mgr.palms[0]
            time.sleep(0.2)
            if self.t == None:
                self.t = threading.Timer(3, self.init_motor)
                self.t.start()
            return
        else:
            if self.t != None:
                self.t.cancel()
                self.t = None

        input_angles = robot_controller.update(self._robot.angles_rad, self._target_pos).input_angles_rad
        self._robot.input_angles(input_angles)
        self.send_angles(input_angles)

        self._robot_visualizer \
            .add_point(self._target_pos[0, 0], self._target_pos[1, 0], self._target_pos[2, 0]) \
            .add_robot(self._robot.joint_pos) \
            .draw()

    def send_angles(self, angles):
        reactor.callFromThread(factory._proto.sendMessage, CMD_W_MOTOR_ANGLE(int(1), int(angles[0, 0]), int(30)))
        reactor.callFromThread(factory._proto.sendMessage, CMD_W_MOTOR_ANGLE(int(2), int(angles[1, 0]), int(20)))
        reactor.callFromThread(factory._proto.sendMessage, CMD_W_MOTOR_ANGLE(int(3), int(angles[2, 0]), int(10)))
        reactor.callFromThread(factory._proto.sendMessage, CMD_W_MOTOR_ANGLE(int(4), int(angles[3, 0]), int(10)))


def arm_ctrl_main_loop(n, t):
    plam_mgr = PlumManager()

    print " === start sub thread (arm_ctrl_main_loop) === "

    while not stop_event.is_set():
        global portstate
        if portstate == 'open':
            plam_mgr.update()
        else:
            time.sleep(1)

    print " === end sub thread (arm_ctrl_main_loop) === "


if __name__ == '__main__':
    # sudo service leapd start

    factory = WebSocketClientFactory("ws://127.0.0.1:9001")

    factory.setSessionParameters(
        None,
        None,
        ['serial-service-protocol'],
        None,
        None,
        None)
    factory.protocol = MyClientProtocol
    iconnector = reactor.connectTCP("127.0.0.1", 9001, factory)
    import atexit

    atexit.register(my_cleanup, 'clean-up')

    th_me = threading.Thread(target=hoge, name="th_me", args=(5, 5,))
    th_me.start()

    th_arm_main_loop = threading.Thread(target=arm_ctrl_main_loop, name="arm_ctrl_main_loop", args=(5, 5,))
    th_arm_main_loop.start()

    reactor.run()

    stop_event.set()
    th_arm_main_loop.join()

    th_me.join()

    print('exit')
