
#ifndef OpenGLHeaderH
#define OpenGLHeaderH

#if defined(KRKRZ_ENABLE_CANVAS) || defined(_WIN32)
#ifdef __EMSCRIPTEN__
#include <EGL/egl.h>
#include <GLES3/gl31.h>
#elif !defined(_WIN32)
#include "OpenGLHeaderSDL2.h"
#else
#ifdef WIN32
#include "OpenGLHeaderWin32.h"
#elif defined( ANDROID )
#include <android/native_window.h>
#include <EGL/egl.h>
#include <GLES/gl.h>
#include "gl3stub.h"
#endif
#endif


extern void TVPInitializeOpenGLPlatform();
#endif


TJS_EXP_FUNC_DEF(void*, TVPeglGetProcAddress, (const char * procname));

#endif
