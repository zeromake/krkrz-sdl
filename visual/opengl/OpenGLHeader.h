
#ifndef OpenGLHeaderH
#define OpenGLHeaderH

#ifdef __EMSCRIPTEN__
#include <EGL/egl.h>
#include <GLES3/gl31.h>
#else
#include "OpenGLHeaderSDL2.h"
#endif
#if 0
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


TJS_EXP_FUNC_DEF(void*, TVPeglGetProcAddress, (const char * procname));

#endif
