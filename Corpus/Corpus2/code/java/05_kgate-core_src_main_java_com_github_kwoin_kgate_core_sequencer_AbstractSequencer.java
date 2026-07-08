package com.github.kwoin.kgate.core.sequencer;

import com.github.kwoin.kgate.core.message.Message;
import com.github.kwoin.kgate.core.session.Session;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nullable;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.SocketException;
import java.util.Arrays;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.concurrent.CountDownLatch;


/**
 * @author P. WILLEMET
 */
public abstract class AbstractSequencer<T extends Message> implements Iterator<T> {


    private final Logger logger = LoggerFactory.getLogger(AbstractSequencer.class);
    protected Session<T> session;
    protected boolean hasNext;
    protected final ByteArrayOutputStream baos = new ByteArrayOutputStream();
    protected @Nullable CountDownLatch oppositeSessionSignal;


    public void setSession(Session<T> session) {

        this.session = session;
        hasNext = !session.getInput().isInputShutdown();

    }

    
    @Override
    public boolean hasNext() {

        hasNext &= !session.getInput().isClosed();
        return hasNext;

    }


    @Override
    @Nullable
    public T next() {

        if(!hasNext())
            throw new NoSuchElementException();

        baos.reset();

        if(oppositeSessionSignal != null) {
            try {
                oppositeSessionSignal.await();
            } catch (InterruptedException e) {
                logger.warn("Waiting for opposite session signal interrupted");
                oppositeSessionSignal = null;
            }
        }

        try {
            return readNextMessage();
        } catch (SocketException e) {
            logger.debug("Input read() interrupted because socket has been closed");
            hasNext = false;
            return null;
        } catch (IOException e) {
            logger.error("Unexpected error while reading next message", e);
            return null;
        } finally {
            resetState();
        }

    }


    protected abstract T readNextMessage() throws IOException;


    protected abstract void resetState();


    protected void waitForOppositeSessionSignal() {

        if(oppositeSessionSignal == null) {
            logger.debug("Wait for opposite session...");
            oppositeSessionSignal = new CountDownLatch(1);
        }

    }


    public void oppositeSessionSignal() {

        if(oppositeSessionSignal != null) {
            logger.debug("wait for opposite session RELEASED");
            oppositeSessionSignal.countDown();
        }

    }


    protected byte readByte() throws IOException {

        int read = session.getInput().getInputStream().read();
        baos.write(read);
        return (byte) read;

    }


    protected byte[] readBytes(int n) throws IOException {

        byte[] bytes = new byte[n];
        for (int i = 0; i < n; i++)
            bytes[i] = readByte();

        return bytes;

    }


    protected byte[] readUntil(byte[] end, boolean withEnd) throws IOException {

        int read;
        int cursor = 0;
        ByteArrayOutputStream tmpBaos = new ByteArrayOutputStream();

        while(cursor < end.length) {

            read = readByte();
            cursor = read == end[cursor] ? cursor + 1 : 0;
            tmpBaos.write(read);

        }

        byte[] bytes = tmpBaos.toByteArray();
        return withEnd ? bytes : Arrays.copyOf(bytes, bytes.length - end.length);

    }


}
