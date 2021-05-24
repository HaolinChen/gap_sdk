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


int test_entry()
{
  while(*(volatile int *)0x1c000000 != 0x11223344)
  {
  }

  printf("Received expected value\n");

  *(volatile int *)0x1c000000 = 0x55667788;

  while(1);

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
