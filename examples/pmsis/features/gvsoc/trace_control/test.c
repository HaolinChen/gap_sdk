/* 
 * Copyright (C) 2021 GreenWaves Technologies
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
 *
 */

#include "pmsis.h"
#include "stdio.h"

#ifdef __PLATFORM_GVSOC__
#include <pmsis/platforms/gvsoc.h>
#endif


int test_entry()
{
  // Activate FC instruction traces and dump to file log
  gv_trace_enable("fc/insn:log");

  printf("(%ld, %ld) Entering main controller\n", pi_cluster_id(), pi_core_id());

  // Deactivate traces
  gv_trace_disable("fc/insn:log");

  return 0;
}

void test_kickoff(void *arg)
{
  int ret = test_entry();
  pmsis_exit(ret);
}

int main()
{
  return pmsis_kickoff((void *)test_kickoff);
}
