import queue


class _LiveBitstreamFeeder:
    """
    Feeds compressed OAK H264/H265 bytes into PyNvVideoCodec CreateDemuxer(callback).

    One feeder is used per encoded stream: RGB, left mono, right mono.
    This is safer than manually creating nvc.PacketData.
    """

    def __init__(self, bitstream_queue, stop_event):
        self.q = bitstream_queue
        self.stop_event = stop_event
        self.pending = bytearray()
        self.eof = False
        self.total_bytes_fed = 0

    def feed_chunk(self, demuxer_buffer):
        # During shutdown, do not feed buffered partial GOP data to the demuxer.
        # Return EOF immediately so PyNvVideoCodec can unwind cleanly.
        if self.stop_event.is_set():
            self.pending.clear()
            self.eof = True
            return 0

        capacity = len(demuxer_buffer)

        while len(self.pending) == 0 and not self.eof:
            if self.stop_event.is_set():
                self.eof = True
                break

            try:
                item = self.q.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                self.eof = True
                break

            self.pending.extend(item)

        if len(self.pending) == 0 and self.eof:
            return 0

        n = min(capacity, len(self.pending))
        demuxer_buffer[:n] = self.pending[:n]
        del self.pending[:n]
        self.total_bytes_fed += n
        return n