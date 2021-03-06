
import tables as tb
import numpy  as np
import pandas as pd

from .. reco            import tbl_functions as tbl
from .. core            import system_of_units as units
from .. core.exceptions import SensorBinningNotFound
from .. core.exceptions import NoParticleInfoInFile

from .. evm.event_model import MCParticle
from .. evm.event_model import MCHit
from .. evm.event_model import Waveform

from .. evm.nh5 import MCGeneratorInfo
from .. evm.nh5 import MCExtentInfo
from .. evm.nh5 import MCHitInfo
from .. evm.nh5 import MCParticleInfo

from .. database import load_db as DB

from typing import Mapping
from typing import Sequence
from typing import Tuple
from typing import List
from typing import Type

# use Mapping (duck type) rather than dict

units_dict = {'picosecond' : units.picosecond,  'ps' : units.picosecond,
              'nanosecond' : units.nanosecond,  'ns' : units.nanosecond,
              'microsecond': units.microsecond, 'mus': units.microsecond,
              'millisecond': units.millisecond, 'ms' : units.millisecond}


class mc_info_writer:
    """Write MC info to file."""
    def __init__(self, h5file, compression = 'ZLIB4'):

        self.h5file      = h5file
        self.compression = compression
        self._create_tables()
        self.reset()
        self.current_tables = None

        self.last_written_hit      = 0
        self.last_written_particle = 0
        self.first_extent_row      = True
        self.first_file            = True


    def reset(self):
        # last visited row
        self.last_row              = 0

    def _create_tables(self):
        """Create tables in MC group in file h5file."""
        if '/MC' in self.h5file:
            MC = self.h5file.root.MC
        else:
            MC = self.h5file.create_group(self.h5file.root, "MC")

        self.extent_table = self.h5file.create_table(MC, "extents",
                                                     description = MCExtentInfo,
                                                     title       = "extents",
                                                     filters     = tbl.filters(self.compression))

        # Mark column to index after populating table
        self.extent_table.set_attr('columns_to_index', ['evt_number'])

        self.hit_table = self.h5file.create_table(MC, "hits",
                                                  description = MCHitInfo,
                                                  title       = "hits",
                                                  filters     = tbl.filters(self.compression))

        self.particle_table = self.h5file.create_table(MC, "particles",
                                                       description = MCParticleInfo,
                                                       title       = "particles",
                                                       filters     = tbl.filters(self.compression))

        self.generator_table = self.h5file.create_table(MC, "generators",
                                                        description = MCGeneratorInfo,
                                                        title       = "generators",
                                                        filters     = tbl.filters(self.compression))



    def __call__(self, mctables: (tb.Table, tb.Table, tb.Table, tb.Table),
                 evt_number: int):
        if mctables is not self.current_tables:
            self.current_tables = mctables
            self.reset()

        extents = mctables[0]
        # Note: filtered out events do not make it to evt_number, but are included in extents dataset
        for iext in range(self.last_row, len(extents)):
            this_row = extents[iext]
            if this_row['evt_number'] == evt_number:
                if iext == 0:
                    if self.first_file:
                        modified_hit          = this_row['last_hit']
                        modified_particle     = this_row['last_particle']
                        self.first_extent_row = False
                        self.first_file       = False
                    else:
                        modified_hit          = this_row['last_hit'] + self.last_written_hit + 1
                        modified_particle     = this_row['last_particle'] + self.last_written_particle + 1
                        self.first_extent_row = False

                elif self.first_extent_row:
                    previous_row          = extents[iext-1]
                    modified_hit          = this_row['last_hit'] - previous_row['last_hit'] + self.last_written_hit - 1
                    modified_particle     = this_row['last_particle'] - previous_row['last_particle'] + self.last_written_particle - 1
                    self.first_extent_row = False
                    self.first_file       = False
                else:
                    previous_row      = extents[iext-1]
                    modified_hit      = this_row['last_hit'] - previous_row['last_hit'] + self.last_written_hit
                    modified_particle = this_row['last_particle'] - previous_row['last_particle'] + self.last_written_particle

                modified_row                  = self.extent_table.row
                modified_row['evt_number']    = evt_number
                modified_row['last_hit']      = modified_hit
                modified_row['last_particle'] = modified_particle
                modified_row.append()

                self.last_written_hit      = modified_hit
                self.last_written_particle = modified_particle

                break

        self.extent_table.flush()

        hits, particles, generators = read_mcinfo_evt(mctables, evt_number, self.last_row)
        self.last_row = iext + 1

        for h in hits:
            new_row = self.hit_table.row
            new_row['hit_position']  = h[0]
            new_row['hit_time']      = h[1]
            new_row['hit_energy']    = h[2]
            new_row['label']         = h[3]
            new_row['particle_indx'] = h[4]
            new_row['hit_indx']      = h[5]
            new_row.append()
        self.hit_table.flush()

        for p in particles:
            new_row = self.particle_table.row
            new_row['particle_indx']  = p[0]
            new_row['particle_name']  = p[1]
            new_row['primary']        = p[2]
            new_row['mother_indx']    = p[3]
            new_row['initial_vertex'] = p[4]
            new_row['final_vertex']   = p[5]
            new_row['initial_volume'] = p[6]
            new_row['final_volume']   = p[7]
            new_row['momentum']       = p[8]
            new_row['kin_energy']     = p[9]
            new_row['creator_proc']   = p[10]
            new_row.append()
        self.particle_table.flush()

        for g in generators:
            new_row = self.generator_table.row
            new_row['evt_number']    = g[0]
            new_row['atomic_number'] = g[1]
            new_row['mass_number']   = g[2]
            new_row['region']        = g[3]
            new_row.append()
        self.generator_table.flush()


def copy_mc_info(h5in : tb.File, writer : Type[mc_info_writer], which_events : List[int]=None):
    """Copy from the input file to the output file the MC info of certain events.

    Parameters
    ----------
    h5in  : tb.File
        Input h5 file.
    writer : instance of class mcinfo_io.mc_info_writer
        MC info writer to h5 file.
    which_events : None or list of ints
        List of IDs (i.e. event numbers) that identify the events to be copied
        to the output file. If None, all events in the input file are copied.
    """
    try:
        if which_events is None:
            which_events = h5in.root.MC.extents.cols.evt_number[:]
        mcinfo = tbl.get_mc_info(h5in)
        for n in which_events:
            writer(mctables=mcinfo, evt_number=n)
    except tb.exceptions.NoSuchNodeError:
        raise tb.exceptions.NoSuchNodeError(f"No MC info in file {h5in}.")
    except IndexError:
        raise IndexError(f"No event {n} in file {h5in}.")
    except NoParticleInfoInFile as err:
        print(f"Warning: {h5in}", err)
        pass


def load_mchits(file_name: str,
                event_range=(0, int(1e9))) -> Mapping[int, Sequence[MCHit]]:

    with tb.open_file(file_name, mode='r') as h5in:
        mchits_dict = read_mchit_info(h5in, event_range)

    return mchits_dict


def load_mchits_df(file_name : str) -> pd.DataFrame:
    """
    Opens file and calls read_mchits_df
    file_name : str
                The name of the file to be read
    """
    extents = pd.read_hdf(file_name, 'MC/extents')
    with tb.open_file(file_name) as h5in:
        hits = read_mchits_df(h5in, extents)

    return hits


def read_mchits_df(h5in    : tb.file.File,
                   extents : pd.DataFrame) -> pd.DataFrame:
    """
    Loads the MC hit information into a pandas DataFrame.
    h5in    : pytables file
              A file already read into a pytables object
    extents : pd.DataFrame
              The extents table from the file which gives
              information about the event tracture.
    """

    hits_tb  = h5in.root.MC.hits

    # Generating hits DataFrame
    hits = pd.DataFrame({'hit_id'      : hits_tb.col('hit_indx'),
                         'particle_id' : hits_tb.col('particle_indx'),
                         'label'       : hits_tb.col('label').astype('U13'),
                         'time'        : hits_tb.col('hit_time'),
                         'x'           : hits_tb.col('hit_position')[:, 0],
                         'y'           : hits_tb.col('hit_position')[:, 1],
                         'z'           : hits_tb.col('hit_position')[:, 2],
                         'energy'      : hits_tb.col('hit_energy')})

    evt_hit_df = extents[['last_hit', 'evt_number']]
    evt_hit_df.set_index('last_hit', inplace = True)

    hits = hits.merge(evt_hit_df          ,
                      left_index  =   True,
                      right_index =   True,
                      how         = 'left')
    hits.rename(columns={"evt_number": "event_id"}, inplace = True)
    hits.event_id.fillna(method='bfill', inplace = True)
    hits.event_id = hits.event_id.astype(int)

    # Setting the indexes
    hits.set_index(['event_id', 'particle_id', 'hit_id'], inplace=True)

    return hits


def load_mcparticles(file_name: str,
                     event_range=(0, int(1e9))) -> Mapping[int, Mapping[int, MCParticle]]:

    with tb.open_file(file_name, mode='r') as h5in:
        return read_mcinfo(h5in, event_range)


def load_mcparticles_df(file_name: str) -> pd.DataFrame:
    """
    Opens file and calls read_mcparticles_df
    file_name : str
                The name of the file to be read
    """
    extents = pd.read_hdf(file_name, 'MC/extents')
    with tb.open_file(file_name, mode='r') as h5in:
        particles = read_mcparticles_df(h5in, extents)

    return particles


def read_mcparticles_df(h5in    : tb.file.File,
                        extents : pd.DataFrame) -> pd.DataFrame:
    """
    A reader for the MC particle output based
    on pandas DataFrames.

    file_name: string
               Name of the file to be read
    """
    p_tb = h5in.root.MC.particles

    # Generating parts DataFrame
    parts = pd.DataFrame({'particle_id'       : p_tb.col('particle_indx'),
                          'particle_name'     : p_tb.col('particle_name').astype('U20'),
                          'primary'           : p_tb.col('primary').astype('bool'),
                          'mother_id'         : p_tb.col('mother_indx'),
                          'initial_x'         : p_tb.col('initial_vertex')[:, 0],
                          'initial_y'         : p_tb.col('initial_vertex')[:, 1],
                          'initial_z'         : p_tb.col('initial_vertex')[:, 2],
                          'initial_t'         : p_tb.col('initial_vertex')[:, 3],
                          'final_x'           : p_tb.col('final_vertex')[:, 0],
                          'final_y'           : p_tb.col('final_vertex')[:, 1],
                          'final_z'           : p_tb.col('final_vertex')[:, 2],
                          'final_t'           : p_tb.col('final_vertex')[:, 3],
                          'initial_volume'    : p_tb.col('initial_volume').astype('U20'),
                          'final_volume'      : p_tb.col('final_volume').astype('U20'),
                          'initial_momentum_x': p_tb.col('momentum')[:, 0],
                          'initial_momentum_y': p_tb.col('momentum')[:, 1],
                          'initial_momentum_z': p_tb.col('momentum')[:, 2],
                          'kin_energy'        : p_tb.col('kin_energy'),
                          'creator_proc'      : p_tb.col('creator_proc').astype('U20')})

    # Adding event info
    evt_part_df = extents[['last_particle', 'evt_number']]
    evt_part_df.set_index('last_particle', inplace = True)
    parts = parts.merge(evt_part_df         ,
                        left_index  =   True,
                        right_index =   True,
                        how         = 'left')
    parts.rename(columns={"evt_number": "event_id"}, inplace = True)
    parts.event_id.fillna(method='bfill', inplace = True)
    parts.event_id = parts.event_id.astype(int)

    # Setting the indexes
    parts.set_index(['event_id', 'particle_id'], inplace=True)

    return parts


def load_mcsensor_response(file_name: str,
                           event_range=(0, int(1e9))) -> Mapping[int, Mapping[int, Waveform]]:

    with tb.open_file(file_name, mode='r') as h5in:
        return read_mcsns_response(h5in, event_range)


def get_sensor_binning(file_name : str) -> Tuple:
    """
    Looks in the configuration table of the
    input file and extracts the binning used
    for both types of sensitive detector.
    """
    config   = pd.read_hdf(file_name, 'MC/configuration')
    bins     = config[config.param_key.str.contains('time_binning')]
    pmt_bin  = bins.param_value[bins.param_key.str.contains('Pmt')].iloc[0]
    pmt_bin  = float(pmt_bin.split()[0]) * units_dict[pmt_bin.split()[1]]
    sipm_bin = bins.param_value[bins.param_key.str.contains('SiPM')].iloc[0]
    sipm_bin = float(sipm_bin.split()[0]) * units_dict[sipm_bin.split()[1]]

    return pmt_bin, sipm_bin


def load_mcsensor_response_df(file_name : str,
                              db_file   : str,
                              run_no    : int) -> Tuple:
    """
    A reader for the MC sensor output based
    on pandas DataFrames.

    file_name: string
               Name of the file to be read
    db_file  : string
               Name of the detector database to be accessed
    run_no   : int
               Run number for database access
    """
    pmt_ids = DB.DataPMT(db_file, run_no).SensorID

    pmt_bin, sipm_bin = get_sensor_binning(file_name)

    extents  = pd.read_hdf(file_name, 'MC/extents')

    sns      = pd.read_hdf(file_name, 'MC/waveforms')
    evt_sns  = extents[['last_sns_data', 'evt_number']]
    evt_sns.set_index('last_sns_data', inplace = True)

    sns = sns.merge(evt_sns             ,
                    left_index  =   True,
                    right_index =   True,
                    how         = 'left')
    sns.evt_number.fillna(method='bfill', inplace = True)

    sns['time'] = sns[sns.sensor_id.isin(pmt_ids)].time_bin * pmt_bin
    sns.time.fillna(sns.time_bin * sipm_bin, inplace = True)

    sns.evt_number = sns.evt_number.astype(int)
    sns.rename(columns = {'evt_number': 'event_id'}, inplace = True)
    sns.set_index(['event_id', 'sensor_id', 'time_bin'], inplace = True)

    return extents.evt_number.unique(), pmt_bin, sipm_bin, sns


def read_mcinfo_evt (mctables: (tb.Table, tb.Table, tb.Table, tb.Table), event_number: int, last_row=0,
                     return_only_hits: bool=False) -> ([tb.Table], [tb.Table], [tb.Table]):
    h5extents    = mctables[0]
    h5hits       = mctables[1]
    h5particles  = mctables[2]
    h5generators = mctables[3]

    particle_rows  = []
    hit_rows       = []
    generator_rows = []

    event_range = (last_row, int(1e9))
    for iext in range(*event_range):
        this_row = h5extents[iext]
        if this_row['evt_number'] == event_number:
            # the indices of the first hit and particle are 0 unless the first event
            #  written is to be skipped: in this case they must be read from the extents
            ihit = ipart = 0
            if iext > 0:
                previous_row = h5extents[iext-1]

                ihit         = int(previous_row['last_hit']) + 1
                if not return_only_hits:
                    ipart        = int(previous_row['last_particle']) + 1

            ihit_end  = this_row['last_hit']
            if len(h5hits) != 0:
                while ihit <= ihit_end:
                    hit_rows.append(h5hits[ihit])
                    ihit += 1

            if return_only_hits: break

            ipart_end = this_row['last_particle']
            while ipart <= ipart_end:
                particle_rows.append(h5particles[ipart])
                ipart += 1

            # It is possible for the 'generators' dataset to be empty. In this case, do not add any rows to 'generators'.
            if len(h5generators) != 0:
                generator_rows.append(h5generators[iext])

            break

    return hit_rows, particle_rows, generator_rows


def read_mcinfo(h5f, event_range=(0, int(1e9))) -> Mapping[int, Mapping[int, Sequence[MCParticle]]]:
    mc_info = tbl.get_mc_info(h5f)

    h5extents = mc_info.extents

    events_in_file = len(h5extents)

    all_events = {}

    for iext in range(*event_range):
        if iext >= events_in_file:
            break

        current_event           = {}
        evt_number              = h5extents[iext]['evt_number']
        hit_rows, particle_rows, generator_rows = read_mcinfo_evt(mc_info, evt_number, iext)

        for h5particle in particle_rows:
            this_particle = h5particle['particle_indx']
            current_event[this_particle] = MCParticle(h5particle['particle_name'].decode('utf-8','ignore'),
                                                      h5particle['primary'],
                                                      h5particle['mother_indx'],
                                                      h5particle['initial_vertex'],
                                                      h5particle['final_vertex'],
                                                      h5particle['initial_volume'].decode('utf-8','ignore'),
                                                      h5particle['final_volume'].decode('utf-8','ignore'),
                                                      h5particle['momentum'],
                                                      h5particle['kin_energy'],
                                                      h5particle['creator_proc'].decode('utf-8','ignore'))

        for h5hit in hit_rows:
            ipart            = h5hit['particle_indx']
            current_particle = current_event[ipart]

            hit = MCHit(h5hit['hit_position'],
                        h5hit['hit_time'],
                        h5hit['hit_energy'],
                        h5hit['label'].decode('utf-8','ignore'))

            current_particle.hits.append(hit)

        evt_number             = h5extents[iext]['evt_number']
        all_events[evt_number] = current_event

    return all_events


def compute_mchits_dict(mcevents:Mapping[int, Mapping[int, MCParticle]]) -> Mapping[int, Sequence[MCHit]]:
    """Returns all hits in the event"""
    mchits_dict = {}
    for event_no, particle_dict in mcevents.items():
        hits = []
        for particle_no in particle_dict.keys():
            particle = particle_dict[particle_no]
            hits.extend(particle.hits)
        mchits_dict[event_no] = hits
    return mchits_dict


def read_mchit_info(h5f, event_range=(0, int(1e9))) -> Mapping[int, Sequence[MCHit]]:
    """Returns all hits in the event"""
    mc_info = tbl.get_mc_info(h5f)
    h5extents = mc_info.extents
    events_in_file = len(h5extents)

    all_events = {}

    for iext in range(*event_range):
        if iext >= events_in_file:
            break

        current_event  = {}
        evt_number     = h5extents[iext]['evt_number']
        hit_rows, _, _ = read_mcinfo_evt(mc_info, evt_number, iext, True)

        hits = []
        for h5hit in hit_rows:
            hit = MCHit(h5hit['hit_position'],
                        h5hit['hit_time'],
                        h5hit['hit_energy'],
                        h5hit['label'].decode('utf-8','ignore'))
            hits.append(hit)

        all_events[evt_number] = hits

    return all_events


def read_mcsns_response(h5f, event_range=(0, 1e9)) -> Mapping[int, Mapping[int, Waveform]]:

    h5config = h5f.root.MC.configuration

    bin_width_PMT  = None
    bin_width_SiPM = None
    for row in h5config:
        param_name = row['param_key'].decode('utf-8','ignore')
        if param_name.find('time_binning') >= 0:
            param_value = row['param_value'].decode('utf-8','ignore')
            numb, unit  = param_value.split()
            if param_name.find('Pmt') > 0:
                bin_width_PMT = float(numb) * units_dict[unit]
            elif param_name.find('SiPM') >= 0:
                bin_width_SiPM = float(numb) * units_dict[unit]


    if bin_width_PMT is None:
        raise SensorBinningNotFound
    if bin_width_SiPM is None:
        raise SensorBinningNotFound


    h5extents   = h5f.root.MC.extents

    try:
        h5f.root.MC.waveforms[0]
    except IndexError:
        print('Error: this file has no sensor response information.')

    h5waveforms = h5f.root.MC.waveforms

    last_line_of_event = 'last_sns_data'
    events_in_file     = len(h5extents)

    all_events = {}

    iwvf = 0
    if event_range[0] > 0:
        iwvf = h5extents[event_range[0]-1][last_line_of_event] + 1

    for iext in range(*event_range):
        if iext >= events_in_file:
            break

        current_event = {}

        iwvf_end          = h5extents[iext][last_line_of_event]
        current_sensor_id = h5waveforms[iwvf]['sensor_id']
        time_bins         = []
        charges           = []
        while iwvf <= iwvf_end:
            wvf_row   = h5waveforms[iwvf]
            sensor_id = wvf_row['sensor_id']

            if sensor_id == current_sensor_id:
                time_bins.append(wvf_row['time_bin'])
                charges.  append(wvf_row['charge'])
            else:
                bin_width = bin_width_PMT if current_sensor_id < 1000 else bin_width_SiPM
                times     = np.array(time_bins) * bin_width

                current_event[current_sensor_id] = Waveform(times, charges, bin_width)

                time_bins = []
                charges   = []
                time_bins.append(wvf_row['time_bin'])
                charges.append(wvf_row['charge'])

                current_sensor_id = sensor_id

            iwvf += 1

        bin_width = bin_width_PMT if current_sensor_id < 1000 else bin_width_SiPM
        times     = np.array(time_bins) * bin_width
        current_event[current_sensor_id] = Waveform(times, charges, bin_width)

        evt_number             = h5extents[iext]['evt_number']
        all_events[evt_number] = current_event

    return all_events
