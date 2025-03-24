"""
    Minimal usage example
    Connects to QTM and streams 3D data forever
    (start QTM first, load file, Play->Play with Real-Time output)
"""

import asyncio
import qtm_rt
import numpy as np

def on_packet(packet):
    global last_meas 
    """ Callback function that is called everytime a data packet arrives from QTM """
    # print("Framenumber: {}".format(packet.framenumber))
    # header, markers = packet.get_3d_markers()
    # print("Component info: {}".format(header))
    # for marker in markers:
    #     print("\t", marker)
    _, bodies = packet.get_6d()
    for i, body in enumerate(bodies):
        pos, _ = body
        x, y, z = pos
        if i == 0:
            last_meas = np.array([x,y,z]) * 1e-3
            if last_meas is not None:
                print(last_meas)
        # print(f"Body n {i}, Pos = {x}, {y}, {z}")


async def setup():
    """ Main function """
    connection = await qtm_rt.connect("192.168.225.1")
    if connection is None:
        return

    await connection.stream_frames(components=["6d"], on_packet=on_packet)


async def main():
    asyncio.create_task(setup())

    await asyncio.sleep(1)
    data = []

    for _ in range(1000): 
        if last_meas is not None:
            data.append(last_meas.copy())
        await asyncio.sleep(0.01)
        # time.sleep(0.01)

    np.savez_compressed('data/mocap_example.npz', pos=np.asarray(data))
    print('Data saved')

# if __name__ == "__main__":
#     asyncio.ensure_future(setup())
#     asyncio.get_event_loop().run_forever()

asyncio.run(main())
print('Finish acquisition')

import matplotlib.pyplot as plt
data = np.load('data/mocap_example.npz')

print(data['pos'].shape)

ax = plt.figure().add_subplot(projection='3d')
ax.plot(data['pos'][:, 0], data['pos'][:, 1], data['pos'][:, 2])

ax.set_xlabel('x')
ax.set_ylabel('y')

plt.show()
for i in range(len(data['pos'])):
    print(data['pos'][i])
