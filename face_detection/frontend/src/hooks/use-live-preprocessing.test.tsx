import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useLivePreprocessing } from "./use-live-preprocessing";
import type { PreprocessResponse } from "../lib/live-detection";

function makeResponse(
  faces: PreprocessResponse["faces"] = [],
): PreprocessResponse {
  return {
    face_count: faces.length,
    image_dimensions: { width: 640, height: 480 },
    quality_score: 81,
    quality_flags: faces.length === 0 ? ["no_face_detected"] : [],
    quality_metrics: { blur_variance: 175.2 },
    faces,
  };
}

describe("useLivePreprocessing", () => {
  let video: HTMLVideoElement;
  let captureCanvas: HTMLCanvasElement;
  let videoRef: { current: HTMLVideoElement };
  let captureCanvasRef: { current: HTMLCanvasElement };
  let track: MediaStreamTrack & { stop: ReturnType<typeof vi.fn> };
  let stream: MediaStream;
  let getUserMedia: ReturnType<typeof vi.fn>;
  let fetchMock: ReturnType<typeof vi.fn>;
  let canvasContext: { drawImage: ReturnType<typeof vi.fn>; clearRect: ReturnType<typeof vi.fn> };
  let toBlobMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.useRealTimers();
    video = document.createElement("video");
    captureCanvas = document.createElement("canvas");
    Object.defineProperty(video, "videoWidth", { configurable: true, value: 640 });
    Object.defineProperty(video, "videoHeight", { configurable: true, value: 480 });
    Object.defineProperty(video, "play", {
      configurable: true,
      value: vi.fn().mockResolvedValue(undefined),
    });
    Object.defineProperty(video, "pause", {
      configurable: true,
      value: vi.fn(),
    });

    canvasContext = {
      drawImage: vi.fn(),
      clearRect: vi.fn(),
    };
    Object.defineProperty(captureCanvas, "getContext", {
      configurable: true,
      value: vi.fn(() => canvasContext),
    });
    toBlobMock = vi.fn((callback: BlobCallback) =>
      callback(new Blob(["test-frame"], { type: "image/jpeg" })),
    );
    Object.defineProperty(captureCanvas, "toBlob", {
      configurable: true,
      value: toBlobMock,
    });
    videoRef = { current: video };
    captureCanvasRef = { current: captureCanvas };

    track = {
      stop: vi.fn(),
      onended: null,
    } as unknown as MediaStreamTrack & { stop: ReturnType<typeof vi.fn> };
    stream = {
      getTracks: () => [track],
      getVideoTracks: () => [track],
    } as unknown as MediaStream;
    getUserMedia = vi.fn().mockResolvedValue(stream);
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: { getUserMedia },
    });

    fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: vi.fn().mockResolvedValue(makeResponse()),
    });
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("requests the camera only after start and captures a multipart frame", async () => {
    const { result } = renderHook(() =>
      useLivePreprocessing(videoRef, captureCanvasRef),
    );

    expect(getUserMedia).not.toHaveBeenCalled();
    await act(async () => {
      await result.current.startCamera();
    });
    await waitFor(() => expect(fetchMock).toHaveBeenCalledOnce());

    expect(result.current.cameraStatus).toBe("Active");
    expect(getUserMedia).toHaveBeenCalledWith({
      audio: false,
      video: { facingMode: "user" },
    });
    expect(video.srcObject).toBe(stream);
    const options = fetchMock.mock.calls[0][1] as RequestInit;
    expect(options.method).toBe("POST");
    expect(options.body).toBeInstanceOf(FormData);
    expect((options.body as FormData).get("file")).toBeInstanceOf(File);
    expect(options.headers).toBeUndefined();
    expect(canvasContext.drawImage).toHaveBeenCalledWith(
      video,
      0,
      0,
      640,
      480,
    );
    expect(toBlobMock).toHaveBeenCalledWith(expect.any(Function), "image/jpeg", 0.85);
  });

  it("prevents a second camera session while initialization is pending", async () => {
    let resolveStream: ((value: MediaStream) => void) | undefined;
    getUserMedia.mockReturnValue(
      new Promise((resolve) => {
        resolveStream = resolve;
      }),
    );
    const { result } = renderHook(() =>
      useLivePreprocessing(videoRef, captureCanvasRef),
    );
    let startPromise: Promise<void> | undefined;
    act(() => {
      startPromise = result.current.startCamera();
    });
    act(() => {
      void result.current.startCamera();
    });
    expect(getUserMedia).toHaveBeenCalledOnce();

    await act(async () => {
      resolveStream?.(stream);
      await startPromise;
    });
    expect(result.current.cameraStatus).toBe("Active");
  });

  it("reports camera permission denial", async () => {
    getUserMedia.mockRejectedValue(
      new DOMException("Permission denied", "NotAllowedError"),
    );
    const { result } = renderHook(() =>
      useLivePreprocessing(videoRef, captureCanvasRef),
    );

    await act(async () => {
      await result.current.startCamera();
    });

    expect(result.current.cameraStatus).toBe("Error");
    expect(result.current.errorMessage).toContain("permission was denied");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("reports a camera disconnection and releases its stream", async () => {
    const { result } = renderHook(() =>
      useLivePreprocessing(videoRef, captureCanvasRef),
    );
    await act(async () => {
      await result.current.startCamera();
    });

    act(() => {
      track.onended?.call(track, new Event("ended"));
    });

    expect(result.current.cameraStatus).toBe("Error");
    expect(result.current.errorMessage).toContain("camera disconnected");
    expect(track.stop).toHaveBeenCalledOnce();
    expect(video.srcObject).toBeNull();
  });

  it("stops camera tracks and aborts active preprocessing on stop", async () => {
    let rejectRequest: ((reason: DOMException) => void) | undefined;
    fetchMock.mockImplementation(
      (_url: string, options: RequestInit) =>
        new Promise((_resolve, reject) => {
          options.signal?.addEventListener("abort", () => {
            rejectRequest = reject;
            reject(new DOMException("Aborted", "AbortError"));
          });
        }),
    );
    const { result } = renderHook(() =>
      useLivePreprocessing(videoRef, captureCanvasRef),
    );
    await act(async () => {
      await result.current.startCamera();
    });
    await waitFor(() => expect(fetchMock).toHaveBeenCalledOnce());
    const signal = (fetchMock.mock.calls[0][1] as RequestInit).signal;

    act(() => result.current.stopCamera());

    expect(signal?.aborted).toBe(true);
    expect(track.stop).toHaveBeenCalledOnce();
    expect(video.srcObject).toBeNull();
    expect(result.current.response).toBeNull();
    expect(rejectRequest).toBeDefined();
    await waitFor(() => expect(result.current.cameraStatus).toBe("Idle"));
  });

  it("shows actual zero-face and multiple-face backend responses", async () => {
    const multiFace = makeResponse([
      {
        face_id: 0,
        detection_index: 0,
        bounding_box: { x: 10, y: 20, width: 120, height: 130 },
        confidence: 0.88,
        quality_flags: [],
      },
      {
        face_id: 1,
        detection_index: 1,
        bounding_box: { x: 300, y: 25, width: 90, height: 100 },
        confidence: 0.79,
        quality_flags: ["small_face"],
      },
    ]);
    fetchMock.mockResolvedValue({
      ok: true,
      json: vi.fn().mockResolvedValueOnce(makeResponse()).mockResolvedValue(multiFace),
    });
    const { result } = renderHook(() =>
      useLivePreprocessing(videoRef, captureCanvasRef),
    );
    await act(async () => {
      await result.current.startCamera();
    });
    await waitFor(() => expect(result.current.response?.face_count).toBe(0));
    expect(result.current.response?.quality_flags).toContain("no_face_detected");

    act(() => result.current.stopCamera());
    await waitFor(() => expect(result.current.cameraStatus).toBe("Idle"));
    fetchMock.mockResolvedValue({
      ok: true,
      json: vi.fn().mockResolvedValue(multiFace),
    });
    await act(async () => {
      await result.current.startCamera();
    });
    await waitFor(() => expect(result.current.response?.face_count).toBe(2));
    expect(
      result.current.response?.faces.map((face) => face.detection_index),
    ).toEqual([0, 1]);
  });

  it("reports an unavailable backend without stopping the camera", async () => {
    fetchMock.mockRejectedValue(new TypeError("Failed to fetch"));
    const { result } = renderHook(() =>
      useLivePreprocessing(videoRef, captureCanvasRef),
    );
    await act(async () => {
      await result.current.startCamera();
    });
    await waitFor(() =>
      expect(result.current.backendStatus).toBe("Unavailable"),
    );

    expect(result.current.cameraStatus).toBe("Active");
    expect(result.current.errorMessage).toContain("Could not reach");
  });

  it("times out an unresponsive backend request", async () => {
    vi.useFakeTimers();
    fetchMock.mockImplementation(
      (_url: string, options: RequestInit) =>
        new Promise((_resolve, reject) => {
          options.signal?.addEventListener("abort", () => {
            reject(new DOMException("Aborted", "AbortError"));
          });
        }),
    );
    const { result } = renderHook(() =>
      useLivePreprocessing(videoRef, captureCanvasRef),
    );
    await act(async () => {
      await result.current.startCamera();
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(15_000);
    });

    expect(result.current.backendStatus).toBe("Unavailable");
    expect(result.current.errorMessage).toContain("timed out");
    act(() => result.current.stopCamera());
  });

  it("does not start another inference while one request is pending", async () => {
    vi.useFakeTimers();
    fetchMock.mockImplementation(
      (_url: string, options: RequestInit) =>
        new Promise((_resolve, reject) => {
          options.signal?.addEventListener("abort", () => {
            reject(new DOMException("Aborted", "AbortError"));
          });
        }),
    );
    const { result } = renderHook(() =>
      useLivePreprocessing(videoRef, captureCanvasRef),
    );
    await act(async () => {
      await result.current.startCamera();
    });
    expect(fetchMock).toHaveBeenCalledOnce();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(4_000);
    });
    expect(fetchMock).toHaveBeenCalledOnce();
    act(() => result.current.stopCamera());
  });

  it("ignores a stale response after stopping and releases tracks on unmount", async () => {
    let resolveRequest: ((value: Response) => void) | undefined;
    fetchMock.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveRequest = resolve;
        }),
    );
    const { result, unmount } = renderHook(() =>
      useLivePreprocessing(videoRef, captureCanvasRef),
    );
    await act(async () => {
      await result.current.startCamera();
    });
    await waitFor(() => expect(resolveRequest).toBeDefined());

    act(() => result.current.stopCamera());
    await act(async () => {
      resolveRequest?.({
        ok: true,
        json: async () => makeResponse(),
      } as Response);
      await Promise.resolve();
    });
    expect(result.current.response).toBeNull();

    await waitFor(() => expect(result.current.cameraStatus).toBe("Idle"));
    await act(async () => {
      await result.current.startCamera();
    });
    unmount();
    expect(track.stop).toHaveBeenCalled();
    expect(video.srcObject).toBeNull();
  });
});
