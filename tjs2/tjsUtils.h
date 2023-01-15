//---------------------------------------------------------------------------
/*
	TJS2 Script Engine
	Copyright (C) 2000 W.Dee <dee@kikyou.info> and contributors

	See details of license at "license.txt"
*/
//---------------------------------------------------------------------------
// utility functions
//---------------------------------------------------------------------------
#ifndef tjsUtilsH
#define tjsUtilsH

#include "tjsVariant.h"
#include "tjsString.h"

#if 1
#ifdef KRKRZ_USE_SDL_THREADS
#include <SDL.h>
#elif defined(_WIN32)
#include <windows.h>
#include <mutex>
#else
#if (!defined(__EMSCRIPTEN__)) || (defined(__EMSCRIPTEN__) && defined(__EMSCRIPTEN_PTHREADS__))
#include <mutex>
#endif
#endif
#else
#ifdef __WIN32__
#include <windows.h>
#else
#include <semaphore.h>
#endif
#endif

#include <thread>

namespace TJS
{
//---------------------------------------------------------------------------
// tTJSCriticalSection ( implement on each platform for multi-threading support )
//---------------------------------------------------------------------------
#if 1
class tTJSCriticalSection
{
#ifdef KRKRZ_USE_SDL_THREADS
	SDL_mutex *Mutex;
#else
#if (!defined(__EMSCRIPTEN__)) || (defined(__EMSCRIPTEN__) && defined(__EMSCRIPTEN_PTHREADS__))
	std::recursive_mutex Mutex;
#endif
#endif

public:
	tTJSCriticalSection()
	{
#ifdef KRKRZ_USE_SDL_THREADS
		Mutex = SDL_CreateMutex();
#endif
	}
	~tTJSCriticalSection()
	{
#ifdef KRKRZ_USE_SDL_THREADS
		SDL_DestroyMutex(Mutex);
#endif
	}

#ifdef KRKRZ_USE_SDL_THREADS
	void Enter()
	{
		SDL_LockMutex(Mutex);
	}
	void Leave()
	{
		SDL_UnlockMutex(Mutex);
	}
#else
#if (!defined(__EMSCRIPTEN__)) || (defined(__EMSCRIPTEN__) && defined(__EMSCRIPTEN_PTHREADS__))
	void Enter() { Mutex.lock(); }
	void Leave() { Mutex.unlock(); }
#else
	void Enter() {}
	void Leave() {}
#endif
#endif
};
#else
#ifdef __WIN32__
class tTJSCriticalSection
{
	CRITICAL_SECTION CS;
public:
	tTJSCriticalSection() { InitializeCriticalSection(&CS); }
	~tTJSCriticalSection() { DeleteCriticalSection(&CS); }

	void Enter() { EnterCriticalSection(&CS); }
	void Leave() { LeaveCriticalSection(&CS); }
};
#else
// implements Semaphore
class tTJSCriticalSection
{
	sem_t Handle;
public:
	tTJSCriticalSection() { sem_init( &Handle, 0, 1 ); }
	~tTJSCriticalSection() { sem_destroy( &Handle ); }

	void Enter() { sem_wait( &Handle ); }
	void Leave() { sem_post( &Handle ); }
};
#endif
#endif
//---------------------------------------------------------------------------
// interlocked operation ( implement on each platform for multi-threading support )
//---------------------------------------------------------------------------
// refer C++11 atomic

//---------------------------------------------------------------------------
// tTJSCriticalSectionHolder
//---------------------------------------------------------------------------
class tTJSCriticalSectionHolder
{
	tTJSCriticalSection *Section;
public:
	tTJSCriticalSectionHolder(tTJSCriticalSection &cs)
	{
		Section = &cs;
		Section->Enter();
	}

	~tTJSCriticalSectionHolder()
	{
		Section->Leave();
	}

};
typedef tTJSCriticalSectionHolder tTJSCSH;
//---------------------------------------------------------------------------

struct tTJSSpinLock {
	std::atomic_flag atom_lock;
	tTJSSpinLock();
	void lock(); // will stuck if locked in same thread!
	void unlock();
};

class tTJSSpinLockHolder {
	tTJSSpinLock* Lock;
public:
	tTJSSpinLockHolder(tTJSSpinLock &lock);

	~tTJSSpinLockHolder();
};

//---------------------------------------------------------------------------
// tTJSAtExit / tTJSAtStart
//---------------------------------------------------------------------------
class tTJSAtExit
{
	void (*Function)();
public:
	tTJSAtExit(void (*func)()) { Function = func; };
	~tTJSAtExit() { Function(); }
};
//---------------------------------------------------------------------------
class tTJSAtStart
{
public:
	tTJSAtStart(void (*func)()) { func(); };
};
//---------------------------------------------------------------------------
class iTJSDispatch2;
extern iTJSDispatch2 * TJSObjectTraceTarget;

#define TJS_DEBUG_REFERENCE_BREAK \
	if(TJSObjectTraceTarget == (iTJSDispatch2*)this) TJSNativeDebuggerBreak()
#define TJS_SET_REFERENCE_BREAK(x) TJSObjectTraceTarget=(x)
//---------------------------------------------------------------------------
TJS_EXP_FUNC_DEF(const tjs_char *, TJSVariantTypeToTypeString, (tTJSVariantType type));
	// convert given variant type to type string ( "void", "int", "object" etc.)

TJS_EXP_FUNC_DEF(tTJSString, TJSVariantToReadableString, (const tTJSVariant &val, tjs_int maxlen = 512));
	// convert given variant to human-readable string
	// ( eg. "(string)\"this is a\\nstring\"" )
TJS_EXP_FUNC_DEF(tTJSString, TJSVariantToExpressionString, (const tTJSVariant &val));
	// convert given variant to string which can be interpret as an expression.
	// this function does not convert objects ( returns empty string )

//---------------------------------------------------------------------------



/*[*/
//---------------------------------------------------------------------------
// tTJSRefHolder : a object holder for classes that has AddRef and Release methods
//---------------------------------------------------------------------------
template <typename T>
class tTJSRefHolder
{
private:
	T *Object;
public:
	tTJSRefHolder(T * ref) { Object = ref; Object->AddRef(); }
	tTJSRefHolder(const tTJSRefHolder<T> &ref)
	{
		Object = ref.Object;
		Object->AddRef();
	}
	~tTJSRefHolder() { Object->Release(); }

	T* GetObject() { Object->AddRef(); return Object; }
	T* GetObjectNoAddRef() { return Object; }

	const tTJSRefHolder & operator = (const tTJSRefHolder & rhs)
	{
		if(rhs.Object != Object)
		{
			Object->Release();
			Object = rhs.Object;
			Object->AddRef();
		}
		return *this;
	}
};



/*]*/

//---------------------------------------------------------------------------



//---------------------------------------------------------------------------
// TJSAlignedAlloc : aligned memory allocater
//---------------------------------------------------------------------------
TJS_EXP_FUNC_DEF(void *, TJSAlignedAlloc, (tjs_uint bytes, tjs_uint align_bits));
TJS_EXP_FUNC_DEF(void, TJSAlignedDealloc, (void *ptr));
//---------------------------------------------------------------------------

/*[*/
//---------------------------------------------------------------------------
// floating-point class checker
//---------------------------------------------------------------------------
// constants used in TJSGetFPClass
#define TJS_FC_CLASS_MASK 7
#define TJS_FC_SIGN_MASK 8

#define TJS_FC_CLASS_NORMAL 0
#define TJS_FC_CLASS_NAN 1
#define TJS_FC_CLASS_INF 2

#define TJS_FC_IS_NORMAL(x)  (((x)&TJS_FC_CLASS_MASK) == TJS_FC_CLASS_NORMAL)
#define TJS_FC_IS_NAN(x)  (((x)&TJS_FC_CLASS_MASK) == TJS_FC_CLASS_NAN)
#define TJS_FC_IS_INF(x)  (((x)&TJS_FC_CLASS_MASK) == TJS_FC_CLASS_INF)

#define TJS_FC_IS_NEGATIVE(x) (0!=((x) & TJS_FC_SIGN_MASK))
#define TJS_FC_IS_POSITIVE(x) (!TJS_FC_IS_NEGATIVE(x))


/*]*/
TJS_EXP_FUNC_DEF(tjs_uint32, TJSGetFPClass, (tjs_real r));
//---------------------------------------------------------------------------
}

#endif




