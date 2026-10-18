// Lithe co-scheduling for ROCr (lithe-docker docs/PLAN-gpu-and-network-locality.md 4.2, 4.2a).
//
// All references to parlib / liblithe are weak: in a process that does not link them every helper
// reports "not cooperative" and ROCr behaves exactly as upstream. In a lithified process, a caller
// that is a Lithe context and may yield (parlib_can_cooperatively_yield(), GAPS B13: constructors,
// TSD destructors and _dl_fini may not) gets:
//   - LitheYield() instead of sched_yield / a bare spin (signal waits, os::YieldThread),
//   - LitheBlocking(f): the kernel wait in f runs on a parlib helper kernel thread while the context
//     is parked on an eventfd (parlib_blocking_call), so the hart runs other contexts,
//   - Lithe progress contexts instead of pthreads from os::CreateThread.
// pthread_create is not intercepted anywhere: ROCr's own creation site chooses.
//   ROCR_LITHE=0 disables all of it at run time.
#ifndef HSA_RUNTIME_CORE_UTIL_LITHE_COOP_H_
#define HSA_RUNTIME_CORE_UTIL_LITHE_COOP_H_

#include <cstddef>
#include <cstdlib>

extern "C" {
int parlib_can_cooperatively_yield(void) __attribute__((weak));
int parlib_blocking_call(void (*fn)(void*), void* arg) __attribute__((weak));
void lithe_context_yield(void) __attribute__((weak));
void* lithe_sched_current(void) __attribute__((weak));
void* lithe_fork_join_context_create_progress(size_t stack_size, void (*start)(void*), void* arg)
    __attribute__((weak));
}

namespace rocr {
namespace os {

inline bool LitheEnabled() {
  static int on = -1;
  if (on < 0) {
    const char* e = getenv("ROCR_LITHE");
    on = (e && *e) ? (atoi(e) != 0) : 1;
  }
  return on && parlib_can_cooperatively_yield && lithe_context_yield && lithe_sched_current;
}

// True when the caller is a Lithe context that may yield right now.
inline bool LitheCoop() {
  return LitheEnabled() && lithe_sched_current() != nullptr && parlib_can_cooperatively_yield();
}

inline void LitheYield() { lithe_context_yield(); }

// Run f() (a kernel-blocking call) without holding the caller's hart when cooperative.
template <class F> inline void LitheBlocking(F&& f) {
  if (LitheCoop() && parlib_blocking_call) {
    struct Box {
      F* f;
      static void Run(void* p) { (*static_cast<Box*>(p)->f)(); }
    } box{&f};
    parlib_blocking_call(&Box::Run, &box);
  } else {
    f();
  }
}

}  // namespace os
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_UTIL_LITHE_COOP_H_
