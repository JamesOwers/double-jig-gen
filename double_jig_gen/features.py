import json
import logging

import music21
import pandas as pd

LOGGER = logging.getLogger(__name__)


def get_score_features(score: music21.stream.Score):
    try:
        parts = score.parts
    except Exception as e:
        return dict(nr_parts=None, error=f"Error getting parts: {e}")
    nr_parts = len(parts)
    if nr_parts > 1:
        return dict(nr_parts=nr_parts)
    part = parts[0]
    try:
        part_features = get_part_features(part)
    except Exception as e:
        part_features = dict(error=f"Error getting part features: {e}")
    return dict(
        nr_parts=nr_parts,
        **part_features,
    )


def get_part_features(part: music21.stream.Part, return_objects: bool = False):
    part_objects = dict(
        time_signatures=part.getTimeSignatures(
            returnDefault=False, sortByCreationTime=True
        ),
        keys=part[music21.key.Key],
        measures=part[music21.stream.Measure],
        has_chords=part.hasElementOfClass(music21.chord.Chord),
    )
    nr_measures = len(part_objects["measures"])
    if not nr_measures > 0:
        return dict(nr_measures=nr_measures)
    key_features = get_key_features(part_objects["keys"])
    time_signature_features = get_time_signature_features(
        part_objects["time_signatures"]
    )
    sections = get_section_information(part)
    section_features = get_section_features(sections)
    out_dict = dict(
        nr_measures=nr_measures,
        **key_features,
        **time_signature_features,
        **section_features,
    )
    if return_objects:
        out_dict = {**out_dict, **part_objects, **sections}
    return out_dict


def get_key_features(keys):
    nr_keys = len(keys)
    if nr_keys > 0:
        first_key = keys[0]
        return dict(
            nr_keys=nr_keys,
            key_name=first_key.name,
            key_nr_sharps=first_key.sharps,
        )
    else:
        return dict(nr_keys=nr_keys)


def get_time_signature_features(time_signatures):
    nr_ts = len(time_signatures)
    if nr_ts > 0:
        first_ts = time_signatures[0]
        ts = first_ts
        return dict(
            nr_ts=nr_ts,
            ts_str=ts.ratioString,
            ts_numerator=ts.numerator,
            ts_denominator=ts.denominator,
        )
    else:
        return dict(nr_ts=nr_ts)


def get_section_information(part):
    def write_section(sections, section, how_many):
        if (len(section["measures"]) != 0) and (how_many is not None):
            section["how_many"] = how_many
            sections.append(section)
        section = {"measures": [], "measure_repeat_nr": [], "how_many": None}
        return sections, section

    def append_measure(section, measure, measure_repeat_nr=0):
        if (
            len(
                measure.getElementsByClass(
                    [music21.note.Note, music21.note.Rest, music21.chord.Chord]
                )
            )
            == 0
        ):
            # don't append, the measure is empty
            return section
        section["measures"].append(measure)
        section["measure_repeat_nr"].append(measure_repeat_nr)
        return section

    def has_barline(measure, side, must_be_repeat):
        if side == "left":
            bar = measure.leftBarline
        elif side == "right":
            bar = measure.rightBarline
        else:
            raise ValueError()
        if bar:
            if must_be_repeat:
                return isinstance(bar, music21.bar.Repeat)
            else:
                return True
        else:
            return False

    repeat_measures = {}
    sections = []
    section = {"measures": [], "measure_repeat_nr": [], "how_many": None}
    previous_bar_ended_section = False
    ignored_classes = (
        music21.spanner.Slur,
        music21.key.Key,
        #         music21.clef.TrebleClef,
        #         music21.meter.TimeSignature,
        #         music21.chord.Chord,
        #         music21.note.Note,
    )
    how_many = None
    must_be_repeat = False
    for obj in part:
        if isinstance(obj, music21.spanner.RepeatBracket):
            measure_repeat_nr = int(
                obj.number
            )  # this will fail if it's not an int, see .getNumberList()
            for measure in obj.getSpannedElements():
                repeat_measures[measure] = measure_repeat_nr
        elif isinstance(obj, music21.stream.Measure):
            measure = obj
            if measure in repeat_measures:
                measure_repeat_nr = repeat_measures[measure]
                how_many = measure_repeat_nr
                del repeat_measures[measure]
                if len(repeat_measures) == 0:
                    previous_bar_ended_section = True
                elif has_barline(measure, side="right", must_be_repeat=False):
                    previous_bar_ended_section = True
                section = append_measure(section, measure, measure_repeat_nr)
            elif has_barline(measure, side="right", must_be_repeat=must_be_repeat):
                #                 if not measure.rightBarline.type in ["double", "final"]:
                #                     print(measure.rightBarline.type)
                #                     print(measure.rightBarline)
                # TODO: handle case where previous_bar_ended_section is already true
                previous_bar_ended_section = True
                if isinstance(measure.rightBarline, music21.bar.Repeat):
                    rb_times = measure.rightBarline.times
                    if rb_times is None:
                        rb_times = 2
                else:
                    rb_times = None
                if (how_many is None or how_many == 1) and rb_times is not None:
                    how_many = rb_times
                section = append_measure(section, measure, measure_repeat_nr=0)
            elif (
                has_barline(measure, side="left", must_be_repeat=must_be_repeat)
                or previous_bar_ended_section
            ):
                previous_bar_ended_section = False
                sections, section = write_section(sections, section, how_many)
                section = append_measure(section, measure, measure_repeat_nr=0)
                how_many = None
                if isinstance(measure.leftBarline, music21.bar.Repeat):
                    must_be_repeat = True
                else:
                    must_be_repeat = False
            else:
                previous_bar_ended_section = False
                if how_many is None:
                    how_many = 1
                section = append_measure(section, measure, measure_repeat_nr=0)
        elif isinstance(obj, ignored_classes):
            pass
        else:
            raise (ValueError(f"Not expecting {type(obj)}: {obj}"))
    else:
        sections, section = write_section(sections, section, how_many)
    if len(sections) == 0:
        LOGGER.warning(f"{part} returned nothing:\n{part.show('text')}")
        sections = [{"measures": [], "measure_repeat_nr": [], "how_many": None}]
    return sections


def get_section_features(sections):
    nr_written_measures = sum(len(section["measures"]) for section in sections)
    try:
        first_section = sections[0]
    except Exception as e:
        print(sections)
        raise e
    if len(first_section["measures"]) == 1:
        has_pickup = True
        first_section_idx = 1
    else:
        has_pickup = False
        first_section_idx = 0
    form_str = ",".join(
        [
            str(idx)
            for idx, section in enumerate(sections[first_section_idx:])
            for _ in range(section["how_many"])
        ]
    )
    section_info = []
    for section in sections[first_section_idx:]:
        first_measure = section["measures"][0]
        has_anacrusis = first_measure.duration != first_measure.barDuration
        repeat_numbers = section["measure_repeat_nr"]
        try:
            section_length = (
                max(idx for idx, val in enumerate(repeat_numbers) if val == 1) + 1
            )
        except ValueError:
            section_length = len(section["measures"])
        if has_anacrusis:
            final_measure = section["measures"][-1]
            final_measure_is_balanced = (
                final_measure.duration.quarterLength
                + first_measure.duration.quarterLength
            ) == first_measure.barDuration.quarterLength
            first_repeat_final_measure = section["measures"][section_length - 1]
            first_repeat_final_measure_is_balanced = (
                first_repeat_final_measure.duration.quarterLength
                + first_measure.duration.quarterLength
            ) == first_measure.barDuration.quarterLength
            anacrusis_balance = (
                int(final_measure_is_balanced)
                + int(first_repeat_final_measure_is_balanced)
            ) / 2
            section_length -= 1
        else:
            anacrusis_balance = None
        section_len_is_divisible_by_4 = section_length % 4 == 0
        section_info.append(
            dict(
                has_anacrusis=has_anacrusis,
                anacrusis_balance=anacrusis_balance,
                section_length=section_length,
                section_len_is_divisible_by_4=section_len_is_divisible_by_4,
            )
        )
    section_info_df = pd.DataFrame(section_info)
    mean_anacrusis_balance = section_info_df["anacrusis_balance"].mean()
    mean_section_len_is_divisible_by_4 = section_info_df[
        "section_len_is_divisible_by_4"
    ].mean()
    return dict(
        nr_written_measures=nr_written_measures,
        has_pickup=has_pickup,
        form_str=form_str,
        mean_anacrusis_balance=mean_anacrusis_balance,
        mean_section_len_is_divisible_by_4=mean_section_len_is_divisible_by_4,
        section_info=json.dumps(section_info),
    )
