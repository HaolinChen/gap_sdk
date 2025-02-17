/*
 * Copyright (C) 2020  GreenWaves Technologies, SAS
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

/* 
 * Authors: Germain Haugou, GreenWaves Technologies (germain.haugou@greenwaves-technologies.com)
 */

#include <vp/vp.hpp>
#include <vp/itf/io.hpp>
#include <vp/itf/wire.hpp>
#include <stdio.h>
#include <string.h>

#include "vp/itf/clock.hpp"

#include "archi/chips/gap9_v2/apb_soc_ctrl/apb_soc_ctrl_regs.h"
#include "archi/chips/gap9_v2/apb_soc_ctrl/apb_soc_ctrl_regfields.h"
#include "archi/chips/gap9_v2/apb_soc_ctrl/apb_soc_ctrl_gvsoc.h"

using namespace std::placeholders;

#define L2_NB_BANKS 16
#define L1_NB_BANKS 2

class apb_soc_ctrl : public vp::component
{

public:
    apb_soc_ctrl(js::config *config);

    int build();
    void start();
    void reset(bool active);

    static vp::io_req_status_e req(void *__this, vp::io_req *req);

private:
    //vp::io_req_status_e sleep_ctrl_req(int reg_offset, int size, bool is_write, uint8_t *data);
    //vp::io_req_status_e l2_btrim_stdby_req(int reg_offset, int size, bool is_write, uint8_t *data);
    vp::io_req_status_e fll_ctrl_req(int reg_offset, int size, bool is_write, uint8_t *data);
    vp::io_req_status_e l1_pwr_ctrl_req(int reg_offset, int size, bool is_write, uint8_t *data);
    vp::io_req_status_e l2_pwr_ctrl_req(int reg_offset, int size, bool is_write, uint8_t *data);
    void reg_osc_ctrl_req(uint64_t reg_offset, int size, uint8_t *value, bool is_write);
    void dbg_ctrl_req(uint64_t reg_offset, int size, uint8_t *value, bool is_write);
    void sleep_ctrl_info_req(uint64_t reg_offset, int size, uint8_t *value, bool is_write);
    void sleep_ctrl_req(uint64_t reg_offset, int size, uint8_t *value, bool is_write);
    void clk_div_ref_fast_pow2_req(uint64_t reg_offset, int size, uint8_t *value, bool is_write);
    void l2_pwr_active_req(uint64_t reg_offset, int size, uint8_t *value, bool is_write);

    static void ref_clock_sync(void *__this, bool value);
    static void fast_clock_sync(void *__this, bool value);

    static void bootsel_sync(void *__this, int value);
    static void confreg_ext_sync(void *__this, uint32_t value);
    static void wakeup_rtc_sync(void *__this, bool wakeup);
    static void wakeup_gpio_sync(void *__this, int value, int gpio);
    void set_wakeup(int value);
    void sync_dm_available(uint32_t mask);

    vp::trace trace;
    vp::trace info;
    vp::io_slave in;

    vp::wire_slave<int> bootsel_itf;
    vp::wire_master<uint32_t> bootaddr_itf;
    vp::wire_master<bool> cluster_reset_itf;
    vp::wire_master<bool> cluster_power_itf;
    vp::wire_master<bool> cluster_power_irq_itf;
    vp::wire_master<bool> cluster_clock_gate_irq_itf;
    vp::wire_master<int> event_itf;
    vp::wire_master<bool> ref_clk_ctrl_itf;
    vp::wire_master<bool> fast_clk_ctrl_itf;
    vp::wire_slave<bool> wakeup_rtc_itf;
    vp::wire_master<bool> wakeup_out_itf;
    vp::wire_master<unsigned int> wakeup_seq_itf;
    std::vector<vp::wire_master<bool>> dm_hart_available_itf;
    std::vector<vp::wire_master<bool> *> l2_power_ctrl_itfs;

    std::vector<vp::wire_slave<int>> wakeup_gpio_itf;

    vp::wire_master<uint32_t> confreg_soc_itf;
    vp::wire_slave<uint32_t> confreg_ext_itf;

    int cluster_power_event;
    int cluster_clock_gate_event;

    uint32_t core_status;
    uint32_t bootaddr;
    uint32_t pmu_bypass;
    bool cluster_reset;
    bool cluster_power;
    bool cluster_clock_gate;
    uint32_t l2_vddme;
    uint32_t l2_vddde;
    uint32_t l2_standby;
    uint32_t l1_power;
    uint32_t l1_standby;
    int nb_harts;
    int fast_clock_div_counter;
    int fast_clock_divided_value;
    int nb_l2_shared_banks;

    unsigned int extwake_sync;

    //vp_apb_soc_ctrl_sleep_ctrl r_sleep_ctrl;
    //vp_apb_soc_ctrl_safe_l2_btrim_stdby  r_l2_btrim_stdby;
    //vp_apb_soc_ctrl_safe_fll_ctrl        r_fll_ctrl;
    //vp_apb_soc_ctrl_safe_l1_pwr_ctrl     r_l1_pwr_ctrl;
    //vp_apb_soc_ctrl_safe_l2_pwr_ctrl     r_l2_pwr_ctrl;

    vp::reg_32 jtag_reg_ext;

    vp_regmap_apb_soc_ctrl regmap;

    int wakeup;

    vp::clock_slave ref_clock_itf;
    vp::clock_slave fast_clock_itf;
    vp::clock_master ref_clock_muxed_itf;
};

apb_soc_ctrl::apb_soc_ctrl(js::config *config)
    : vp::component(config)
{
}

void apb_soc_ctrl::set_wakeup(int value)
{
    if (this->wakeup == 0)
    {
        this->wakeup = value;
        this->wakeup_out_itf.sync(value);
    }
}

void apb_soc_ctrl::ref_clock_sync(void *__this, bool value)
{
    apb_soc_ctrl *_this = (apb_soc_ctrl *)__this;

    if (_this->regmap.ref_clk_mux.get() == 0)
    {
        _this->ref_clock_muxed_itf.sync(value);
    }
}

void apb_soc_ctrl::fast_clock_sync(void *__this, bool value)
{
    apb_soc_ctrl *_this = (apb_soc_ctrl *)__this;
    int limit = 1 << _this->regmap.clk_div_ref_fast_pow2.divider_get();
    bool reached = true;

    _this->trace.msg(vp::trace::LEVEL_TRACE, "Updating counter due to fast clock raising edge (counter: %d, limit: %d)\n", _this->fast_clock_div_counter, limit);

    if (_this->regmap.clk_div_ref_fast_pow2.en_get())
    {
        _this->fast_clock_div_counter++;
        if (_this->fast_clock_div_counter == limit)
        {
            _this->fast_clock_div_counter = 0;
        }
        else
        {
            reached = false;
        }
    }

    if (reached)
    {
        _this->trace.msg(vp::trace::LEVEL_TRACE, "Reached limit, raising fast divided edge (value: %d)\n", _this->fast_clock_divided_value);

        if (_this->regmap.ref_clk_mux.get())
        {
            _this->trace.msg(vp::trace::LEVEL_TRACE, "Propagating to muxed ref clock\n");

            _this->ref_clock_muxed_itf.sync(_this->fast_clock_divided_value);
            _this->fast_clock_divided_value ^= 1;
        }
    }
}

#if 0
vp::io_req_status_e apb_soc_ctrl::l2_btrim_stdby_req(int reg_offset, int size, bool is_write, uint8_t *data)
{
  this->r_l2_btrim_stdby.access(reg_offset, size, data, is_write);

  if (is_write)
  {
    this->trace.msg("Modified SAFE_L2_BTRIM_STDBY (btrim: 0x%x, stdby_n: 0x%x)\n",
      this->r_l2_btrim_stdby.btrim_get(),
      this->r_l2_btrim_stdby.stdby_n_get()
    );

    uint32_t standby = ~this->r_l2_btrim_stdby.stdby_n_get();

    for (int i=0; i<L2_NB_BANKS; i++)
    {
      int new_power = (standby >> i) & 1;
      int current_power = (this->l2_standby >> i) & 1;

      if (new_power != current_power)
      {
        if (new_power)
        {
          this->info.msg("Activating cut standby (cut: %d)\n", i);
        }
        else
        {
          this->info.msg("Deactivating cut standby (cut: %d)\n", i);
        }
      }
    }

    this->l2_standby = standby;
  }

  return vp::IO_REQ_OK;
}



vp::io_req_status_e apb_soc_ctrl::fll_ctrl_req(int reg_offset, int size, bool is_write, uint8_t *data)
{
  this->r_fll_ctrl.access(reg_offset, size, data, is_write);

  if (is_write)
  {
    this->trace.msg("Modified SAFE_FLL_CTRL (fll2_pwd: %d, fll3_pwd: %d, fll2_rstb: %d, fll3_rstb: %d)\n",
      this->r_fll_ctrl.fll2_pwd_get(),
      this->r_fll_ctrl.fll3_pwd_get(),
      this->r_fll_ctrl.fll2_rstb_get(),
      this->r_fll_ctrl.fll3_rstb_get()
    );
  }

  return vp::IO_REQ_OK;
}



vp::io_req_status_e apb_soc_ctrl::l1_pwr_ctrl_req(int reg_offset, int size, bool is_write, uint8_t *data)
{
  this->r_l1_pwr_ctrl.access(reg_offset, size, data, is_write);

  if (is_write)
  {
    this->trace.msg("Modified SAFE_L1_PWR_CTRL (btrim: 0x%x, stdby_n: 0x%x, pwd: 0x%x)\n",
      this->r_l1_pwr_ctrl.btrim_get(),
      this->r_l1_pwr_ctrl.stdby_n_get(),
      this->r_l1_pwr_ctrl.pwd_get()
    );

    uint32_t l1_power = ~this->r_l1_pwr_ctrl.pwd_get();

    for (int i=0; i<L1_NB_BANKS; i++)
    {
      int new_power = (l1_power >> i) & 1;
      int current_power = (this->l1_power >> i) & 1;

      if (new_power != current_power)
      {
        if (new_power)
        {
          this->info.msg("Powering-up L1 cut memory array (cut: %d)\n", i);
        }
        else
        {
          this->info.msg("Powering-down L1 cut memory array (cut: %d)\n", i);
        }
      }
    }
    this->l1_power = l1_power;


    uint32_t standby = ~this->r_l1_pwr_ctrl.stdby_n_get();

    for (int i=0; i<L1_NB_BANKS; i++)
    {
      int new_power = (standby >> i) & 1;
      int current_power = (this->l1_standby >> i) & 1;

      if (new_power != current_power)
      {
        if (new_power)
        {
          this->info.msg("Activating L1 cut standby (cut: %d)\n", i);
        }
        else
        {
          this->info.msg("Deactivating L1 cut standby (cut: %d)\n", i);
        }
      }
    }

    this->l1_standby = standby;

  }

  return vp::IO_REQ_OK;
}



vp::io_req_status_e apb_soc_ctrl::l2_pwr_ctrl_req(int reg_offset, int size, bool is_write, uint8_t *data)
{
  this->r_l2_pwr_ctrl.access(reg_offset, size, data, is_write);

  if (is_write)
  {
    this->trace.msg("Modified SAFE_L2_PWR_CTRL (vddde_n: 0x%x, vddme_n: 0x%x)\n",
      this->r_l2_pwr_ctrl.vddde_n_get(),
      this->r_l2_pwr_ctrl.vddme_n_get()
    );

    uint32_t vddme = ~this->r_l2_pwr_ctrl.vddme_n_get();

    for (int i=0; i<L2_NB_BANKS; i++)
    {
      int new_power = (vddme >> i) & 1;
      int current_power = (this->l2_vddme >> i) & 1;

      if (new_power != current_power)
      {
        if (new_power)
        {
          this->info.msg("Powering-up cut memory array (cut: %d)\n", i);
        }
        else
        {
          this->info.msg("Powering-down cut memory array (cut: %d)\n", i);
        }
      }
    }

    this->l2_vddme = vddme;

    uint32_t vddde = ~this->r_l2_pwr_ctrl.vddde_n_get();

    for (int i=0; i<L2_NB_BANKS; i++)
    {
      int new_power = (vddde >> i) & 1;
      int current_power = (this->l2_vddde >> i) & 1;

      if (new_power != current_power)
      {
        if (new_power)
        {
          this->info.msg("Powering-up cut periphery (cut: %d)\n", i);
        }
        else
        {
          this->info.msg("Powering-down cut periphery (cut: %d)\n", i);
        }
      }
    }

    this->l2_vddde = vddde;
  }

  return vp::IO_REQ_OK;
}
#endif


void apb_soc_ctrl::sleep_ctrl_info_req(uint64_t reg_offset, int size, uint8_t *value, bool is_write)
{
    if (!is_write)
    {
        this->regmap.sleep_ctrl_info.set_32(this->regmap.sleep_ctrl.get_32());
        this->regmap.sleep_ctrl_info.update(reg_offset, size, value, is_write);
    }
}


void apb_soc_ctrl::sleep_ctrl_req(uint64_t reg_offset, int size, uint8_t *value, bool is_write)
{
    this->regmap.sleep_ctrl.update(reg_offset, size, value, is_write);

#if 0
    //this->r_sleep_ctrl.access(reg_offset, size, data, is_write);

    if (is_write)
    {
        // this->trace.msg("Modified SLEEP_CTRL (reboot: %d, smartwake_en: %d, rtcwake_en: %d, extwake_type: %d, extwake_en: %d, mram_wakestate: %d, cluster_wakestate: %d, ret_mem: 0x%4.4x)\n",
        //   this->r_sleep_ctrl.reboot_get(),
        //   this->r_sleep_ctrl.smartwake_en_get(),
        //   this->r_sleep_ctrl.rtcwake_en_get(),
        //   this->r_sleep_ctrl.extwake_type_get(),
        //   this->r_sleep_ctrl.extwake_en_get(),
        //   this->r_sleep_ctrl.mram_wakestate_get(),
        //   this->r_sleep_ctrl.cluster_wakestate_get(),
        //   this->r_sleep_ctrl.ret_mem_get()
        // );
    }
    else
    {
        this->trace.msg("Clearing wakeup signal\n");

        this->wakeup = 0;
    }

    return vp::IO_REQ_OK;

#endif

#if 0
    if (is_write)
    {
      uint32_t value = *(uint32_t *)data;
      _this->extwake_sel = (value >> 6) & 0x1f;   // GPIO selection for wakeup
      _this->extwake_type = (value >> 11) & 0x3;  // GPIO wakeup type (raising edge, etc)
      _this->extwake_en = (value >> 13) & 0x1;    // GPIO wakeup enabled
      _this->cfg_wakeup = (value >> 14) & 0x3;    // PMU sequence used to wakeup
      _this->boot_type = (value >> 18) & 0x3;
      _this->wakeup_seq_itf.sync(_this->cfg_wakeup);
    }
    else
    {
      _this->set_wakeup(0);
      *(uint32_t *)data = (_this->extwake_sel << 6) | (_this->extwake_type << 11) | (_this->extwake_en << 13) | (_this->cfg_wakeup << 14) | (_this->boot_type << 18);
    }
#endif
}

vp::io_req_status_e apb_soc_ctrl::req(void *__this, vp::io_req *req)
{
    apb_soc_ctrl *_this = (apb_soc_ctrl *)__this;

    uint64_t offset = req->get_addr();
    uint8_t *data = req->get_data();
    uint64_t size = req->get_size();
    bool is_write = req->get_is_write();

    vp::io_req_status_e err = vp::IO_REQ_OK;

    _this->trace.msg("Apb_soc_ctrl access (offset: 0x%x, size: 0x%x, is_write: %d)\n", offset, size, is_write);

    int reg_id = offset / 4;
    int reg_offset = offset % 4;

    _this->regmap.access(offset, size, data, is_write);

    if (offset == APB_SOC_CTRL_CORESTATUS_OFFSET)
    {
        if (!is_write)
        {
            *(uint32_t *)data = _this->core_status;
        }
        else
        {
            // We are writing to the status register, the 31 LSBs are the return value of the platform and the last bit
            // makes the platform exit when set to 1
            _this->core_status = *(uint32_t *)data;

            if ((_this->core_status >> APB_SOC_CTRL_CORESTATUS_EOC_BIT) & 1)
            {
                _this->clock->stop_engine(_this->core_status & 0x7fffffff);
            }
            else
            {
                uint32_t value = *(uint32_t *)data;
                if (value == 0x0BBAABBA)
                {
                    _this->power.get_engine()->start_capture();
                }
                else if (value == 0x0BBADEAD)
                {
                    _this->power.get_engine()->stop_capture();
                }
            }
        }
    }
    else if (offset == APB_SOC_CTRL_JTAGREG_OFFSET)
    {
        if (is_write)
        {
            _this->confreg_soc_itf.sync(*(uint32_t *)data);
        }
        else
        {
            *(uint32_t *)data = _this->jtag_reg_ext.get() << APB_SOC_CTRL_JTAGREG_EXTERNAL_BIT;
        }
    }
#if 0
  else if (offset == APB_SOC_SAFE_L2_BTRIM_STDBY_OFFSET)
  {
    err = _this->l2_btrim_stdby_req(reg_offset, size, is_write, data);
  }
  else if (offset == APB_SOC_SAFE_FLL_CTRL_OFFSET)
  {
    err = _this->fll_ctrl_req(reg_offset, size, is_write, data);
  }
  else if (offset == APB_SOC_SAFE_L1_PWR_CTRL_OFFSET)
  {
    err = _this->l1_pwr_ctrl_req(reg_offset, size, is_write, data);
  }
  else if (offset == APB_SOC_SAFE_L2_PWR_CTRL_OFFSET)
  {
    err = _this->l2_pwr_ctrl_req(reg_offset, size, is_write, data);
  }
#endif
    else if (offset == APB_SOC_CTRL_FC_BOOT_OFFSET)
    {
        if (is_write)
        {
            _this->trace.msg("Setting boot address (addr: 0x%x)\n", *(uint32_t *)data);
            if (_this->bootaddr_itf.is_bound())
                _this->bootaddr_itf.sync(*(uint32_t *)data);

            _this->bootaddr = *(uint32_t *)data;
        }
        else
            *(uint32_t *)data = _this->bootaddr;
    }
#if 0
  else if (offset == APB_SOC_BYPASS_OFFSET)
  {
    if (is_write)
    {
      _this->trace.msg("Setting PMU bypass (addr: 0x%x)\n", *(uint32_t *)data);

      _this->pmu_bypass = *(uint32_t *)data;

      bool new_cluster_power = (_this->pmu_bypass >> 3) & 1;
      bool new_cluster_reset = (_this->pmu_bypass >> 13) & 1;
      bool new_cluster_clock_gate = (_this->pmu_bypass >> 10) & 1;

      if (_this->cluster_reset != new_cluster_reset)
      {
        if (_this->cluster_reset_itf.is_bound())
        {
          _this->cluster_reset_itf.sync(~new_cluster_reset);
        }

        _this->cluster_reset = new_cluster_reset;
      }

      if (_this->cluster_power != new_cluster_power)
      {
        _this->trace.msg("Setting cluster power (power: 0x%d)\n", new_cluster_power);

        if (_this->cluster_power_itf.is_bound())
        {
          _this->cluster_power_itf.sync(new_cluster_power);
        }

        _this->trace.msg("Triggering soc event (event: 0x%d)\n", _this->cluster_power_event);
        _this->event_itf.sync(_this->cluster_power_event);

        if (_this->cluster_power_irq_itf.is_bound())
        {
          _this->cluster_power_irq_itf.sync(true);
        }
      }

      if (_this->cluster_clock_gate != new_cluster_clock_gate)
      {
        _this->trace.msg("Triggering soc event (event: 0x%d)\n", _this->cluster_clock_gate_event);
        _this->event_itf.sync(_this->cluster_clock_gate_event);

        if (_this->cluster_clock_gate_irq_itf.is_bound())
        {
          _this->cluster_clock_gate_irq_itf.sync(true);
        }
      }

      _this->cluster_power = new_cluster_power;
      _this->cluster_clock_gate = new_cluster_clock_gate;

    }
    else
    {
      *(uint32_t *)data = _this->pmu_bypass;
    }
  }
  else if (offset == APB_SOC_BYPASS_OFFSET)
  {
  }
#endif
    else
    {
    }

    return err;
}

void apb_soc_ctrl::bootsel_sync(void *__this, int value)
{
    apb_soc_ctrl *_this = (apb_soc_ctrl *)__this;
    _this->regmap.bootsel.bootsel_set(value);
}

void apb_soc_ctrl::wakeup_gpio_sync(void *__this, int value, int gpio)
{
#if 0
  apb_soc_ctrl *_this = (apb_soc_ctrl *)__this;
  if (_this->r_sleep_ctrl.extwake_en_get())
  {
    int old_value = _this->extwake_sync;

    _this->extwake_sync = value;

    switch (_this->r_sleep_ctrl.extwake_type_get())
    {
      case 0: {
        if (old_value == 0 && _this->extwake_sync == 1)
          _this->set_wakeup(1);
        break;
      }
      case 1: {
        if (old_value == 1 && _this->extwake_sync == 0)
          _this->set_wakeup(1);
        break;
      }
      case 2: {
        if (_this->extwake_sync == 1)
          _this->set_wakeup(1);
        break;
      }
      case 3: {
        if (_this->extwake_sync == 0)
          _this->set_wakeup(1);
        break;
      }
    }
  }
#endif
}

void apb_soc_ctrl::wakeup_rtc_sync(void *__this, bool wakeup)
{
    apb_soc_ctrl *_this = (apb_soc_ctrl *)__this;
    _this->trace.msg(vp::trace::LEVEL_DEBUG, "Received RTC wakeup\n");
    _this->wakeup_out_itf.sync(1);
}

void apb_soc_ctrl::confreg_ext_sync(void *__this, uint32_t value)
{
    apb_soc_ctrl *_this = (apb_soc_ctrl *)__this;
    _this->jtag_reg_ext.set(value);
}

int apb_soc_ctrl::build()
{
    traces.new_trace("trace", &trace, vp::DEBUG);
    traces.new_trace("info", &info, vp::INFO);
    in.set_req_meth(&apb_soc_ctrl::req);
    new_slave_port("input", &in);

    new_master_port("bootaddr", &this->bootaddr_itf);

    new_master_port("event", &event_itf);

    new_master_port("ref_clk_ctrl", &ref_clk_ctrl_itf);
    new_master_port("fast_clk_ctrl", &fast_clk_ctrl_itf);

    new_master_port("cluster_power", &cluster_power_itf);
    new_master_port("cluster_reset", &cluster_reset_itf);
    new_master_port("cluster_power_irq", &cluster_power_irq_itf);

    new_master_port("cluster_clock_gate_irq", &cluster_clock_gate_irq_itf);

    this->wakeup_rtc_itf.set_sync_meth(&apb_soc_ctrl::wakeup_rtc_sync);
    new_slave_port("wakeup_rtc", &this->wakeup_rtc_itf);

    this->wakeup_gpio_itf.resize(1);
    for (int i = 0; i < 1; i++)
    {
        this->wakeup_gpio_itf[i].set_sync_meth_muxed(&apb_soc_ctrl::wakeup_gpio_sync, i);
        new_slave_port("wakeup_gpio" + std::to_string(i), &this->wakeup_gpio_itf[i]);
    }

    new_master_port("wakeup_out", &this->wakeup_out_itf);

    new_master_port("wakeup_seq", &this->wakeup_seq_itf);

    confreg_ext_itf.set_sync_meth(&apb_soc_ctrl::confreg_ext_sync);
    this->new_slave_port("confreg_ext", &this->confreg_ext_itf);

    this->new_master_port("confreg_soc", &this->confreg_soc_itf);

    this->new_reg("jtag_reg_ext", &this->jtag_reg_ext, 0, false);

    bootsel_itf.set_sync_meth(&apb_soc_ctrl::bootsel_sync);
    new_slave_port("bootsel", &bootsel_itf);

    cluster_power_event = this->get_js_config()->get("cluster_power_event")->get_int();
    cluster_clock_gate_event = this->get_js_config()->get("cluster_clock_gate_event")->get_int();

    core_status = 0;
    this->jtag_reg_ext.set(0);

    // This one is in the always-on domain and so it is reset only when the
    // component is powered-up
    this->wakeup = 0;

    this->extwake_sync = 0;

    this->nb_harts = this->get_js_config()->get_int("nb_harts");
    this->dm_hart_available_itf.resize(this->nb_harts);
    for (int i = 0; i < this->nb_harts; i++)
    {
        this->new_master_port("dm_hart_available_" + std::to_string(i), &this->dm_hart_available_itf[i]);
    }

    //this->new_reg("sleep_ctrl", &this->r_sleep_ctrl, 0, false);
    //this->new_reg("l2_btrim_stdby", &this->r_l2_btrim_stdby, 0, false);
    //this->new_reg("fll_ctrl", &this->r_fll_ctrl, 0, false);
    //this->new_reg("l1_pwr_ctr", &this->r_l1_pwr_ctrl, 0, false);
    //this->new_reg("l2_pwr_ctrl", &this->r_l2_pwr_ctrl, 0, false);

    //this->r_sleep_ctrl.set(0);
    //this->r_l1_pwr_ctrl.set(0xFFFF0);
    //this->r_l2_btrim_stdby.set(0x30);

    this->l2_vddme = (1 << L2_NB_BANKS) - 1;
    this->l2_vddde = (1 << L2_NB_BANKS) - 1;
    this->l2_standby = 0;

    this->l1_power = (1 << L1_NB_BANKS) - 1;
    this->l1_standby = 0;

    this->regmap.build(this, &this->trace);
    this->regmap.sleep_ctrl_info.register_callback(std::bind(&apb_soc_ctrl::sleep_ctrl_info_req, this, _1, _2, _3, _4));
    this->regmap.sleep_ctrl.register_callback(std::bind(&apb_soc_ctrl::sleep_ctrl_req, this, _1, _2, _3, _4));
    this->regmap.dbg_ctrl.register_callback(std::bind(&apb_soc_ctrl::dbg_ctrl_req, this, _1, _2, _3, _4));
    this->regmap.reg_osc_ctrl.register_callback(std::bind(&apb_soc_ctrl::reg_osc_ctrl_req, this, _1, _2, _3, _4));
    this->regmap.clk_div_ref_fast_pow2.register_callback(std::bind(&apb_soc_ctrl::clk_div_ref_fast_pow2_req, this, _1, _2, _3, _4));
    this->regmap.l2_pwr_active.register_callback(std::bind(&apb_soc_ctrl::l2_pwr_active_req, this, _1, _2, _3, _4));

    this->fast_clock_itf.set_sync_meth(&apb_soc_ctrl::fast_clock_sync);
    this->new_slave_port("fast_clock", &fast_clock_itf);

    this->ref_clock_itf.set_sync_meth(&apb_soc_ctrl::ref_clock_sync);
    this->new_slave_port("ref_clock", &ref_clock_itf);

    this->new_master_port("ref_clock_muxed", &this->ref_clock_muxed_itf);

    this->nb_l2_shared_banks = this->get_js_config()->get_child_int("nb_l2_shared_banks");
    for (int i=0; i<this->get_js_config()->get_child_int("nb_l2_shared_banks"); i++)
    {
        vp::wire_master<bool> *itf = new vp::wire_master<bool>();
        this->l2_power_ctrl_itfs.push_back(itf);
        this->new_master_port("l2_power_ctrl_" + std::to_string(i), itf);
    }

    return 0;
}

void apb_soc_ctrl::sync_dm_available(uint32_t mask)
{
    for (int i = 0; i < this->nb_harts; i++)
    {
        if (this->dm_hart_available_itf[i].is_bound())
        {
            this->dm_hart_available_itf[i].sync((mask >> i) & 1);
        }
    }
}

void apb_soc_ctrl::reg_osc_ctrl_req(uint64_t reg_offset, int size, uint8_t *value, bool is_write)
{
    this->regmap.reg_osc_ctrl.update(reg_offset, size, value, is_write);

    if (is_write)
    {
        this->fast_clk_ctrl_itf.sync(this->regmap.reg_osc_ctrl.fast_osc_en_get());
        this->ref_clk_ctrl_itf.sync(this->regmap.reg_osc_ctrl.slow_osc_en_get());
    }
}

void apb_soc_ctrl::dbg_ctrl_req(uint64_t reg_offset, int size, uint8_t *value, bool is_write)
{
    this->regmap.dbg_ctrl.update(reg_offset, size, value, is_write);

    if (is_write)
    {
        this->sync_dm_available(this->regmap.dbg_ctrl.get_32());
    }
}

void apb_soc_ctrl::clk_div_ref_fast_pow2_req(uint64_t reg_offset, int size, uint8_t *value, bool is_write)
{
    this->regmap.clk_div_ref_fast_pow2.update(reg_offset, size, value, is_write);

    if (is_write)
    {
        this->fast_clock_div_counter = 0;
    }
}


void apb_soc_ctrl::l2_pwr_active_req(uint64_t reg_offset, int size, uint8_t *value, bool is_write)
{
    this->regmap.l2_pwr_active.update(reg_offset, size, value, is_write);

    if (is_write)
    {
        uint32_t value = this->regmap.l2_pwr_active.get();
        for (int i=0; i<this->nb_l2_shared_banks; i++)
        {
            this->l2_power_ctrl_itfs[i]->sync(!((value >> i) & 1));
        }
    }
}


void apb_soc_ctrl::reset(bool active)
{
    this->regmap.reset(active);

    if (active)
    {
        this->cluster_power = false;
        this->cluster_clock_gate = false;
        this->fast_clock_div_counter = 0;
        this->fast_clock_divided_value = 0;

        this->sync_dm_available(this->regmap.dbg_ctrl.get_32());

        this->fast_clk_ctrl_itf.sync(this->regmap.reg_osc_ctrl.fast_osc_en_get());
        this->ref_clk_ctrl_itf.sync(this->regmap.reg_osc_ctrl.slow_osc_en_get());
    }
}

void apb_soc_ctrl::start()
{
}

extern "C" vp::component *vp_constructor(js::config *config)
{
    return new apb_soc_ctrl(config);
}
