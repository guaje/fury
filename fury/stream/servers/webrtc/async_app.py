import asyncio
import json
import os
import numpy as np
from functools import partial

from aiohttp import web

from aiortc import RTCPeerConnection, RTCSessionDescription

pcs = set()


async def index(request, **kwargs):
    folder = kwargs['folder']
    content = open(os.path.join(folder, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request, **kwargs):
    folder = kwargs['folder']
    js = kwargs['js']
    content = open(os.path.join(folder, "js/%s" % js), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request, **kwargs):
    video = kwargs['video']
    params = await request.json()
    #print(params["sdp"])
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    # open media source
    audio = None

    await pc.setRemoteDescription(offer)
    for t in pc.getTransceivers():
        if t.kind == "audio" and audio:
            pc.addTrack(audio)
        elif t.kind == "video" and video:
            pc.addTrack(video)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def mouse_weel(request, **kwargs):

    params = await request.json()
    deltaY = float(params['deltaY'])
    circular_quequeue = kwargs['circular_quequeue']
    ok = circular_quequeue.enqueue(np.array([1, deltaY, 0, 0], dtype='float64'))
    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {'was_inserted': ok}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


def get_app(RTCServer, folder=None, circular_quequeue=None):
    if folder is None:
        folder = f'{os.path.dirname(__file__)}/www/'

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", partial(index, folder=folder))

    js_files = ['main.js', 'webrtc.js']
    for js in js_files:
        app.router.add_get(
            "/js/%s" % js, partial(javascript, folder=folder, js=js))
    app.router.add_post("/offer", partial(offer, video=RTCServer))

    if circular_quequeue is not None:
        app.router.add_post("/mouse_weel", partial(
            mouse_weel, circular_quequeue=circular_quequeue))

    return app
