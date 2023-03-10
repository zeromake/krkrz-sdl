

#ifndef __USER_EVENT_H__
#define __USER_EVENT_H__

#if 0
#ifdef _WIN32
#define TVP_EV_TIMER_THREAD			(WM_APP + 1)
#define TVP_EV_WAVE_SND_BUF_THREAD	(TVP_EV_TIMER_THREAD + 1)
#define TVP_EV_VSYNC_TIMING_THREAD	(TVP_EV_WAVE_SND_BUF_THREAD + 1)
#define TVP_EV_CONTINUE_LIMIT_THREAD	(TVP_EV_VSYNC_TIMING_THREAD + 1)
#define TVP_EV_DELIVER_EVENTS_DUMMY	(TVP_EV_CONTINUE_LIMIT_THREAD + 1)
#define TVP_EV_KEEP_ALIVE			(TVP_EV_DELIVER_EVENTS_DUMMY + 1)
#define TVP_EV_IMAGE_LOAD_THREAD	(TVP_EV_KEEP_ALIVE + 1)
#define TVP_EV_WINDOW_RELEASE		(TVP_EV_IMAGE_LOAD_THREAD + 1)
#define TVP_EV_DRAW_TIMING_THREAD	(TVP_EV_WINDOW_RELEASE + 1)
#elif defined(ANDROID)
#endif
#endif
#if 1
#define TVP_EV_TIMER_THREAD			(1024 + 1)
#define TVP_EV_WAVE_SND_BUF_THREAD	(TVP_EV_TIMER_THREAD + 1)
#define TVP_EV_VSYNC_TIMING_THREAD	(TVP_EV_WAVE_SND_BUF_THREAD + 1)
#define TVP_EV_CONTINUE_LIMIT_THREAD	(TVP_EV_VSYNC_TIMING_THREAD + 1)
#define TVP_EV_DELIVER_EVENTS_DUMMY	(TVP_EV_CONTINUE_LIMIT_THREAD + 1)
#define TVP_EV_KEEP_ALIVE			(TVP_EV_DELIVER_EVENTS_DUMMY + 1)
#define TVP_EV_IMAGE_LOAD_THREAD	(TVP_EV_KEEP_ALIVE + 1)
#define TVP_EV_WINDOW_RELEASE		(TVP_EV_IMAGE_LOAD_THREAD + 1)
#define TVP_EV_DRAW_TIMING_THREAD	(TVP_EV_WINDOW_RELEASE + 1)
#endif

#endif // __USER_EVENT_H__

