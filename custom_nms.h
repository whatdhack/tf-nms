// custon_nms.h
#ifndef KERNEL_CUSTOM_NMS_H_
#define KERNEL_CUSTOM_NMS_H_

namespace tensorflow {

namespace functor {

template <typename Device, typename T>
struct CustomNmsFunctor {
  void operator()(const Device& d, int size, const T* in1, const T* in2, T* out);
};

}  // namespace functor

}  // namespace tensorflow

#endif //KERNEL_CUSTOM_NMS_H_
