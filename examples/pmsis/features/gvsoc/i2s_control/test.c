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


#define NB_ELEM 32
#define BUFFER_SIZE (NB_ELEM * 2)

static int16_t *buffers[2];

int test_entry()
{
    pi_device_t i2s;

    // We will will samples from slot 0 in memory and also redirect them to TX with the bypass

    // Open I2S for 44100, 16 bits, 1 slot
    struct pi_i2s_conf i2s_conf;
    pi_i2s_conf_init(&i2s_conf);

    i2s_conf.itf = 0;
    i2s_conf.format = PI_I2S_FMT_DATA_FORMAT_I2S;
    i2s_conf.word_size = 16;
    i2s_conf.channels = 1;
    i2s_conf.options = PI_I2S_OPT_TDM | PI_I2S_OPT_EXT_CLK | PI_I2S_OPT_EXT_WS;

    pi_open_from_conf(&i2s, &i2s_conf);

    if (pi_i2s_open(&i2s))
        return -1;

    // Open slot 0 RX
    struct pi_i2s_channel_conf channel_conf;
    pi_i2s_channel_conf_init(&channel_conf);

    channel_conf.options = PI_I2S_OPT_PINGPONG | PI_I2S_OPT_IS_RX | PI_I2S_OPT_ENABLED;
    buffers[0] = pi_l2_malloc(BUFFER_SIZE);
    buffers[1] = pi_l2_malloc(BUFFER_SIZE);
    if (buffers[0] == NULL || buffers[1] == NULL)
    {
        return -1;
    }
    channel_conf.pingpong_buffers[0] = buffers[0];
    channel_conf.pingpong_buffers[1] = buffers[1];

    channel_conf.block_size = BUFFER_SIZE;
    channel_conf.word_size = 16;
    channel_conf.format = PI_I2S_FMT_DATA_FORMAT_I2S | PI_I2S_CH_FMT_DATA_ORDER_MSB;

    // Open slot 0 TX in bypass mode
    if (pi_i2s_channel_conf_set(&i2s, 0, &channel_conf))
        return -1;

    pi_i2s_channel_conf_init(&channel_conf);

    channel_conf.options = PI_I2S_OPT_IS_TX | PI_I2S_OPT_ENABLED | PI_I2S_OPT_LOOPBACK;

    if (pi_i2s_channel_conf_set(&i2s, 0, &channel_conf))
        return -1;

    if (pi_i2s_ioctl(&i2s, PI_I2S_IOCTL_START, NULL))
        return -1;

    // Print the first buffer of samples
    void *read_buffer;
    int size;
    pi_i2s_channel_read(&i2s, 0, &read_buffer, &size);

    for (int i=0; i<NB_ELEM; i++)
    {
        printf("Sample %d: %d\n", i, buffers[0][i]);
    }

    // Then do nothing and let the bypass propagate samples to TX
    while(1)
    {
      pi_time_wait_us(10000000);
    }

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
