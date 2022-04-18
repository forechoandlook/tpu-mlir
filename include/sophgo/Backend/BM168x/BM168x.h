#pragma once
#include "llvm/Support/DynamicLibrary.h"
#include "mlir/IR/Builders.h"

struct cmd_id_node;
typedef struct cmd_id_node CMD_ID_NODE;

typedef enum {
  STORAGE_MODE_1N_FP32    = 0,
  STORAGE_MODE_1N_INT8    = 1,
  STORAGE_MODE_1N_INT16   = 2,
  STORAGE_MODE_2N_INT16   = 3,
  STORAGE_MODE_4N_INT8    = 4,
  STORAGE_MODE_2IC_FP32   = 5,  // special for 2IC weight
  STORAGE_MODE_4N_4IC_4OC = 6,
  STORAGE_MODE_4N_INT16   = 7,
  STORAGE_MODE_UNINITILIZED,
  STORAGE_MODE_END
} TENSOR_STORAGE_MODE;

typedef enum {
  STORE_MODE_1N = 0,
  STORE_MODE_2N = 1,
  STORE_MODE_4N = 2,
} STORE_MODE_T;

#define BM_BINARY_ADD 0
#define BM_BINARY_SUB 1
#define BM_BINARY_MUL 2
#define BM_BINARY_DIV 3
#define BM_BINARY_MAX 4

#define SUBNET_MODE_TPU 0
#define SUBNET_MODE_CPU 1
#define SUBNET_MODE_MERGE 2
#define SUBNET_MODE_SWITCH 3

#define MEM_TYPE_TPU (1 << 0)
#define MEM_TYPE_CPU (1 << 1)
#define MEM_TYPE_ALL (MEM_TYPE_TPU | MEM_TYPE_CPU)

typedef enum {
  DTYPE_FP32 = 0,
  DTYPE_FP16 = 1,
  DTYPE_INT8 = 2,
  DTYPE_UINT8 = 3,
  DTYPE_INT16 = 4,
  DTYPE_UINT16 = 5,
  DTYPE_INT32 = 6,
  DTYPE_UINT32 = 7,
  DTYPE_BFP16 = 8,
  DTYPE_UNKNOWN = -1,
} DATA_TYPE_T;
typedef DATA_TYPE_T bm_data_type_t;

typedef enum {
  ROUND_INF = 0,     // 1.5 -> 2   -1.5 -> -2
  ROUND_UP = 1,      // 1.5 -> 2   -1.5 -> -1
  ROUND_DOWN = 2,    // 1.5 -> 1   -1.5 -> -2
  ROUND_EVEN = 3,    // 1.5 -> 2    2.5 -> 2
  ROUND_ODD = 4,     // 1.5 -> 1    0.5 -> 1
  ROUND_ZERO = 5,    // 1.5 -> 1   -1.5 -> -1
  TRIM_ZERO = 6,     // 1.6 -> 1   -1.6 -> -1
  TRIM_INF = 7,      // 1.4 -> 2   -1.4 -> -2
  TRIM_UP = 8,       // 1.4 -> 2   -1.6 -> -1
  TRIM_DOWN = 9,     // 1.6 -> 1   -1.4 -> -2
} ROUND_MODE_T;
typedef ROUND_MODE_T bm_round_mode_t;

typedef struct bmcompiler_mem_info {
    uint64_t addr;
    uint64_t size;
    uint64_t offset;
} bm_mem_desc_t;
typedef struct bmcompiler_mem_info bm_device_mem_t;

typedef int (*cmodel_init)(int node_idx, unsigned long long global_mem_size);
typedef void (*cmodel_deinit)(int node_idx);
typedef void * (*create_cmd_id_node)();
typedef void (*destroy_cmd_id_node)(void *pid_node);
typedef void (*set_cmd_id_cycle)(void *pid_node, int val);
typedef int (*get_cmd_id_cycle)(void *pid_node);
typedef void (*reset_cmd_id)(void *pid_node);
typedef void (*allow_store_cmd)();
typedef void (*forbid_store_cmd)();
typedef void (*use_atomic_cmodel)();
typedef void (*forbid_atomic_cmodel)();
typedef void *(*get_global_memaddr)(int node_idx);
typedef void (*set_cmd_buffer_ptr)(void *gdma_buffer_ptr, void *bdc_buffer_ptr);
typedef void (*set_total_id_ptr)(uint32_t *gdma_total_id_ptr, uint32_t *bdc_total_id_ptr, void *cmdid_node, void *gdma_group_id_ptr, void *bdc_group_id_ptr, int *cmdid_groupnum);


namespace sophgo {
namespace backend {
class BM168x {

public:
  virtual void init();
  virtual void before_codegen();
  virtual void after_codegen();
  virtual void deinit();
  static BM168x * instance(const llvm::StringRef chip);
  // -------------------------------------------------------------------
  // functions from nodechip
  // -------------------------------------------------------------------
  cmodel_init dl_cmodel_init;
  cmodel_deinit dl_cmodel_deinit;
  create_cmd_id_node dl_create_cmd_id_node;
  destroy_cmd_id_node dl_destroy_cmd_id_node;
  set_cmd_id_cycle dl_set_cmd_id_cycle;
  get_cmd_id_cycle dl_get_cmd_id_cycle;
  reset_cmd_id dl_reset_cmd_id;
  allow_store_cmd dl_allow_store_cmd;
  forbid_store_cmd dl_forbid_store_cmd;
  use_atomic_cmodel dl_use_atomic_cmodel;
  forbid_atomic_cmodel dl_forbid_atomic_cmodel;
  get_global_memaddr dl_get_global_memaddr;
  set_cmd_buffer_ptr dl_set_cmd_buffer_ptr;
  set_total_id_ptr dl_set_total_id_ptr;

  CMD_ID_NODE *get_cmd_id_node() { return (CMD_ID_NODE *)cmdid_node; }
  void *get_gmem_addr(uint64_t addr);
  void *get_gmem_addr(const bm_device_mem_t &mem);
  void bm_memcpy_s2d(const bm_device_mem_t &dst, void *src);
  void bm_memcpy_d2s(void *dst, const bm_device_mem_t &src);
  void value_s2d(mlir::Value v, void *src);
  void value_d2s(mlir::Value v, void *dst);

  // arch info
  virtual uint64_t get_gmem_start() = 0;
  virtual uint64_t get_ctx_start_addr() = 0;
  virtual uint32_t get_bdc_len(int bdc_num, int group_id) = 0;
  virtual uint32_t get_gdma_len(int gdma_num, int group_id) = 0;
  uint64_t get_cmodel_gmem_size();

  static bm_data_type_t getDataType(mlir::Type type);
  static bm_data_type_t getDataType(mlir::Value v);

public:
  std::shared_ptr<std::vector<uint32_t>> bdc_buffer;
  std::shared_ptr<std::vector<uint32_t>> gdma_buffer;
  uint32_t gdma_total_id;
  uint32_t bdc_total_id;
  std::vector<uint32_t> gdma_group_id;
  std::vector<uint32_t> bdc_group_id;
  std::vector<uint32_t> gdma_bytes;
  std::vector<uint32_t> bdc_bytes;
  int cmdid_groupnum;

  static const int64_t ALIGNMENT = 0x1000;

protected:
  virtual void load_functions();
  void set_command_issue_flag(bool value);
  template <typename FPtrTy> FPtrTy CastToFPtr(const char *symbolName);

protected:
  void *cmdid_node;
  void *bdc_node;
  void *gdma_node;
  bool really_issue_command;
  llvm::sys::DynamicLibrary DL;
};

}
}