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
#include <atomic>
#include <mutex>
#include <sched.h>

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

// Lithe locks only (ROCR_LITHE_LOCKS=0 keeps blocking locks while waits stay cooperative; ablation).
inline bool LitheLockCoop() {
  static int on = -1;
  if (on < 0) {
    const char* e = getenv("ROCR_LITHE_LOCKS");
    on = (e && *e) ? (atoi(e) != 0) : 1;
  }
  return on && LitheCoop();
}

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

// Lithe lock: drop-in for std::mutex under std::lock_guard / std::unique_lock. A Lithe context that
// may yield never blocks its hart on it (try, yield, retry); every other caller locks normally.
class LitheMutex {
 public:
  void lock() {
    if (LitheLockCoop()) {
      while (!m_.try_lock()) LitheYield();
      return;
    }
    m_.lock();
  }
  bool try_lock() { return m_.try_lock(); }
  void unlock() { m_.unlock(); }
 private:
  std::mutex m_;
};

// Lithe once: replaces std::call_once where the initializer can wait (and so yield). std::call_once
// blocks later callers on a futex, which would hold their hart while the initializer's context is
// parked. Here later callers yield (Lithe context) or sched_yield (kernel thread) until it is done.
struct LitheOnceFlag {
  std::atomic<int> state{0};  // 0 not run, 1 running, 2 done
};
template <class F> inline void LitheCallOnce(LitheOnceFlag& flag, F&& f) {
  int s = flag.state.load(std::memory_order_acquire);
  if (s == 2) return;
  int expect = 0;
  if (flag.state.compare_exchange_strong(expect, 1, std::memory_order_acq_rel)) {
    f();
    flag.state.store(2, std::memory_order_release);
    return;
  }
  while (flag.state.load(std::memory_order_acquire) != 2) {
    if (LitheLockCoop()) LitheYield();
    else sched_yield();
  }
}

}  // namespace os
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_UTIL_LITHE_COOP_H_
