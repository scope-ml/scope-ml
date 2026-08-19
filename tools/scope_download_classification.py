#!/usr/bin/env python
import argparse
import pathlib
import pandas as pd
import pkg_resources
from scope.surveys.fritz import api
from scope.utils import (
    read_hdf,
    write_hdf,
    read_parquet,
    write_parquet,
    impute_features,
    parse_load_config,
)
import warnings
import numpy as np
from tools.get_features import get_features
from tools.get_quad_ids import get_cone_ids
import os
from datetime import datetime

NUM_PER_PAGE = 500
CHECKPOINT_NUM = 500
BASE_DIR = pathlib.Path.cwd()

config = parse_load_config()

features_catalog = config['kowalski']['collections']['features']
training_set_config = pathlib.Path(config['training']['dataset'])
training_set_path = BASE_DIR / training_set_config


def organize_source_data(src: pd.DataFrame):
    id = src['id']
    ra = src['ra']
    dec = src['dec']

    data_classes = src['classifications']
    cls_list = ''
    prb_list = ''
    vote_list = ''
    for entry in data_classes:
        cls = entry['classification']
        prb = entry['probability']

        try:
            votes = entry['votes']
            sum_votes = int(np.sum([v['vote'] for v in votes]))
        except TypeError:
            sum_votes = 0

        cls_list += cls + ';'  # same format as download from Fritz frontend
        prb_list += str(prb) + ';'
        vote_list += str(sum_votes) + ';'

    cls_list = cls_list[:-1]  # remove trailing semicolon
    prb_list = prb_list[:-1]
    vote_list = vote_list[:-1]

    data_labellers = src['labellers']
    lbl_list = ''
    for entry in data_labellers:
        lbl = entry['id']
        lbl_list += str(lbl) + ';'

    lbl_list = lbl_list[:-1]

    # loop through annotations, checking for periods
    data_annot = src['annotations']
    origin_list = ''
    period_list = ''
    for entry in data_annot:
        annot_origin = entry['origin']
        annot_data = entry['data']
        annot_name = [x for x in annot_data.keys()]

        for n in annot_name:
            # if period is found, add to list
            if n == 'period':
                origin_list += annot_origin + ';'
                period_list += str(annot_data[n]) + ';'

    origin_list = origin_list[:-1]
    period_list = period_list[:-1]

    dct = {}
    (
        dct['obj_id'],
        dct['ra'],
        dct['dec'],
        dct['classification'],
        dct['probability'],
        dct['period_origin'],
        dct['period'],
        dct['labellers'],
        dct['sum_votes'],
    ) = (
        id,
        ra,
        dec,
        cls_list,
        prb_list,
        origin_list,
        period_list,
        lbl_list,
        vote_list,
    )

    return dct


def merge_sources_features(
    sources,
    features_catalog,
    features_limit=1000,
    taxonomy_map='golden_dataset_mapper.json',
    output_dir='fritzDownload',
    output_filename='merged_classifications_features',
    output_format='parquet',
    get_ztf_filters=False,
    impute_missing_features=False,
    update_training_set: bool = False,
    updated_training_set_prefix: str = 'updated_AL',
    min_net_votes: int = 1,
):

    outpath = str(BASE_DIR / output_dir)
    os.makedirs(outpath, exist_ok=True)

    # Drop rows with duplicate obj_ids (keep first instance)
    dup_obj_id = np.sum(sources.duplicated('obj_id'))
    if dup_obj_id > 0:
        print(f'Dropping {dup_obj_id} sources with duplicate obj_ids.')
        sources = sources.drop_duplicates('obj_id').reset_index(drop=True)

    # Open golden dataset mapper
    mapper_dir = os.path.dirname(BASE_DIR)
    mapper_path = os.path.join(mapper_dir, taxonomy_map)
    gold_map = pd.read_json(mapper_path)

    # Drop columns with no equivalent in Fritz
    none_cols = gold_map.loc['fritz_label'] == 'None'
    gold_map = gold_map.drop(columns=none_cols.index[none_cols.values])

    # Manipulate golden_dataset_mapper to flip keys and values
    gold_map = gold_map.transpose()
    gold_map.index.rename('trainingset_label', inplace=True)
    gold_map = gold_map.reset_index(drop=False).set_index('fritz_label')
    gold_dict = gold_map.transpose().to_dict()

    source_dict_list = []

    for _, row in sources.iterrows():
        gold_dict_specific = gold_dict.copy()

        # Initiate dictonary to concatenate with existing dataframe
        source_dict = {}
        obj_id = row['obj_id']
        ra = row['ra']
        dec = row['dec']

        # Assign Fritz object id
        source_dict['obj_id'] = obj_id
        source_dict['ra'] = ra
        source_dict['dec'] = dec

        classifications = []
        completed_classifications = []

        if not row.isna()['classification']:
            classifications = row['classification'].split(';')
            probabilities = row['probability'].split(';')
            labellers = row['labellers'].split(';')
            vote_sums = row['sum_votes'].split(';')

        total_votes = 0
        # Assign given probability values for each Fritz classification
        for i in range(len(classifications)):
            cls = classifications[i]
            vote_sum = int(vote_sums[i]) if vote_sums[i] != '' else 0
            total_votes += vote_sum

            # If updating training set with AL results, only include classifications with net votes >= min_net_votes
            # Otherwise, include all classifications with nonzero probability
            has_enough_votes = (
                vote_sum >= min_net_votes if update_training_set else True
            )
            do_classification = (not update_training_set) | (
                update_training_set & has_enough_votes
            )

            if (cls not in completed_classifications) & (do_classification):
                try:
                    trainingset_label = gold_dict_specific[cls]['trainingset_label']
                    gold_dict_specific.pop(cls)
                    source_dict[trainingset_label] = float(probabilities[i])
                except KeyError:
                    print(f'Key {cls} not in dataset mapper.')
                    continue
                completed_classifications += [cls]

        # Assign zero probability for remaining labels
        for remaining_entry in gold_dict_specific:
            if remaining_entry not in completed_classifications:
                trainingset_label = gold_dict_specific[remaining_entry][
                    'trainingset_label'
                ]
                source_dict[trainingset_label] = 0.0

        # Labellers assigned by user ID
        if labellers == '':
            labellers = None
        source_dict['labellers'] = labellers

        if (total_votes > 0) | (not update_training_set):
            source_dict_list += [source_dict]

    # Create and write dataframe
    expanded_sources = pd.DataFrame(source_dict_list)

    sources_filepath = os.path.join(outpath, f'sources_{output_dir}' + output_format)
    if output_format == '.csv':
        expanded_sources.to_csv(sources_filepath, index=False)
    elif output_format == '.h5':
        write_hdf(expanded_sources, sources_filepath)
    else:
        write_parquet(expanded_sources, sources_filepath)
    print(f'Saved list of {len(expanded_sources)} unique sources.')

    # Query Kowalski
    print(
        f'Getting all ZTF IDs from cone search for {len(expanded_sources)} sources...'
    )
    ztf_and_obj_ids = get_cone_ids(
        expanded_sources['obj_id'].values,
        expanded_sources['ra'].values,
        expanded_sources['dec'].values,
        catalog=features_catalog,
    )
    # Split ids dataframe based on duplicate vs. non-duplicate rows
    # (For duplicate rows, more than one source claims that set of features)
    ztf_and_obj_ids_nodup = ztf_and_obj_ids.drop_duplicates('_id', keep=False)
    ztf_and_obj_ids_dup = ztf_and_obj_ids[ztf_and_obj_ids.duplicated('_id', keep=False)]

    print(f'Found {len(ztf_and_obj_ids)} rows of features - some may be duplicates.')

    print('Getting non-duplicate features...')
    feature_df_nodup, _ = get_features(
        source_ids=ztf_and_obj_ids_nodup['_id'].values.tolist(),
        features_catalog=features_catalog,
        limit_per_query=features_limit,
    )
    print('Getting duplicate features for further analysis...')
    feature_df_dup, _ = get_features(
        source_ids=ztf_and_obj_ids_dup['_id'].values.tolist(),
        features_catalog=features_catalog,
        limit_per_query=features_limit,
    )

    # Combine obj_ids and ztf_ids for non-duplicate rows
    features_obj_ids_nodup = pd.merge(ztf_and_obj_ids_nodup, feature_df_nodup, on='_id')

    print('Finding closest source for each duplicate set of features...')
    feature_df_dup.set_index('_id', inplace=True)
    dup_expanded_sources = (
        pd.merge(ztf_and_obj_ids_dup, expanded_sources, on='obj_id')
        .reset_index()
        .set_index('_id')
    )
    closest_indices = []
    # Decide which source to assign features based on minimum Euclidean distance from stated ra/dec
    for ID in feature_df_dup.index:
        close_sources = dup_expanded_sources.loc[ID]
        distances = np.sqrt(
            (close_sources['ra'] - feature_df_dup.loc[ID, 'ra']) ** 2
            + (close_sources['dec'] - feature_df_dup.loc[ID, 'dec']) ** 2
        )
        closest_index = close_sources.reset_index().loc[np.argmin(distances), 'index']
        closest_indices += [closest_index]

    closest_dup_expanded_sources = (
        dup_expanded_sources.reset_index()
        .loc[np.unique(closest_indices)]
        .drop(['index'], axis=1)
    )

    # Merge on obj_id, use ra and dec from Fritz
    merged_set_nodup = pd.merge(
        expanded_sources,
        features_obj_ids_nodup.drop(['ra', 'dec'], axis=1),
        on='obj_id',
    )
    merged_set_dup = pd.merge(
        closest_dup_expanded_sources,
        feature_df_dup.drop(['ra', 'dec'], axis=1),
        on='_id',
    )
    # Rejoin duplicate and non-duplicate rows
    merged_set = pd.concat([merged_set_nodup, merged_set_dup]).reset_index(drop=True)

    print(f'Merged set of {len(merged_set)} sources, labels and features.')

    # Get ztf filters if specified
    if get_ztf_filters:
        print('Getting ZTF filters...')
        filter_df, _ = get_features(
            source_ids=merged_set['_id'].values.tolist(),
            features_catalog='ZTF_sources_20210401',
            limit_per_query=features_limit,
            projection={'filter': 1},
        )

        merged_set = pd.merge(merged_set, filter_df, on='_id')

    merged_set['ztf_id'] = merged_set['_id']
    merged_set.drop(['_id'], axis=1, inplace=True)

    source_metadata = sources.attrs
    source_metadata.update(feature_df_nodup.attrs)
    merged_set.attrs = source_metadata

    # Make ztf_id last column in dataframe
    ztf_id_col = merged_set.pop('ztf_id')
    merged_set['ztf_id'] = ztf_id_col

    if impute_missing_features:
        merged_set = impute_features(merged_set, self_impute=True)

    if update_training_set:
        output_filename = f'{updated_training_set_prefix}_{output_filename}'
    filepath = os.path.join(outpath, output_filename + output_format)

    if output_format == '.csv':
        merged_set.to_csv(filepath, index=False)
    elif output_format == '.h5':
        warnings.warn(
            'Coordinates and dmdt features get pickled and put in separate datasets, not supported by hdf5 format.'
        )
        coordinates = merged_set['coordinates']
        dmdt = merged_set['dmdt']
        merged_set = merged_set.drop(['coordinates', 'dmdt'], axis=1)
        # Write hdf5 file
        # Caution: coordinates and dmdt features get pickled and put in separate datasets, not supported by hdf5 format.
        # File will be comparable in size to csv
        with pd.HDFStore(filepath, mode='w') as store:
            store.put('df', merged_set, format='table')
            store.put('coordinates', coordinates)
            store.put('dmdt', dmdt)
            store.get_storer('df').attrs.metadata = merged_set.attrs
    else:
        write_parquet(merged_set, filepath)

    return merged_set


def download_classification(
    file: str,
    group_ids: list,
    start: int,
    merge_features: bool,
    features_catalog: str,
    features_limit: int,
    taxonomy_map: str = 'golden_dataset_mapper.json',
    output_dir: str = 'fritzDownload',
    output_filename: str = 'merged_classifications_features',
    output_format: str = 'parquet',
    get_ztf_filters: bool = False,
    impute_missing_features: bool = False,
    update_training_set: bool = False,
    updated_training_set_prefix: str = 'updated_AL',
    min_net_votes: int = 1,
):
    """
    Download labels from Fritz
    :param file: CSV, hdf5 or parquet file containing obj_id column or "parse" to query by group ids (str)
    :param group_ids: target group ids on Fritz for download (list)
    :param start: page number to start downloading data from (int)
    :param merge_features: if True,  query Kowalski for features to merge (bool)
    :param features_catalog: catalog name to query for features (str)
    :param features_limit: maximum number of sources to query for features per loop (int)
    :param output_dir: directory to write output merged features file (str)
    :param output_filename: name of output merged features file (str)
    :param output_format: format of output merged features file (str)
    :param get_ztf_filters: if True, add ZTF filter ID to default features (bool)
    :param impute_missing_features: if True, impute missing features using scope.utils.impute_features (bool)
    :param update_training_set: if downloading an active learning sample, flag to update the training set with the new classification based on votes (bool)
    :param updated_training_set_prefix: prefix to add to updated trainingset file (str)
    :param min_net_votes: Minimum number of net votes [upvotes - downvotes] to keep an active learning classification. Caution: if zero, all classifications of reviewed sources will be added (int)
    """

    dict_list = []

    # Check for appropriate output format
    output_file_extension = os.path.splitext(output_filename)[-1]

    # If parquet, h5 or csv extension specified in output_filename, use that format for saving
    output_format = (
        output_format
        if output_file_extension not in ['.parquet', '.h5', '.csv']
        else output_file_extension
    )

    if output_format in [
        '.parquet',
        'parquet',
        '.Parquet',
        'Parquet',
        '.parq',
        'parq',
        '.PARQUET',
        'PARQUET',
        '.PARQ',
        'PARQ',
    ]:
        output_format = '.parquet'
        print('Using .parquet extension for saved files.')
    elif output_format in ['.h5', 'h5', '.H5', 'H5', '.hdf5', 'hdf5', '.HDF5', 'HDF5']:
        output_format = '.h5'
        print('Using .h5 extension for saved files.')
    elif output_format in ['.csv', 'csv', '.CSV', 'CSV']:
        output_format = '.csv'
        print('Using .csv extension for saved files.')
    else:
        raise ValueError('Output format must be parquet, hdf5 or csv.')

    # If user puts extension in filename, remove it for consistency
    if (
        (output_filename.endswith('.csv'))
        | (output_filename.endswith('.h5'))
        | (output_filename.endswith('.parquet'))
    ):
        output_filename = os.path.splitext(output_filename)[0]

    outpath = str(BASE_DIR / output_dir)
    os.makedirs(outpath, exist_ok=True)

    filename = (
        os.path.splitext(os.path.basename(file))[0] + '_fritzDownload' + output_format
    )  # rename file
    filepath = os.path.join(outpath, filename)

    # Get code version and current date/time for metadata
    code_version = pkg_resources.get_distribution(
        "scope-ml"
    ).version  # scope.__version__
    utcnow = datetime.utcnow()
    start_dt = utcnow.strftime("%Y-%m-%d %H:%M:%S")

    if file in ["parse", 'Parse', 'PARSE']:
        if group_ids is None:
            raise ValueError('Specify group_ids to query Fritz.')
        response = api(
            "GET",
            "/api/sources",
            {
                "group_ids": group_ids,
                "numPerPage": NUM_PER_PAGE,
                "includeLabellers": True,
            },
        )
        source_data = response.json().get("data")

        # determine number of pages
        allMatches = source_data['totalMatches']
        nPerPage = source_data['numPerPage']
        pages = int(np.ceil(allMatches / nPerPage))

        if start != 0:
            filename = (
                os.path.splitext(filename)[0]
                + f'_continued_page_{start}'
                + output_format
            )
            filepath = os.path.join(outpath, filename)
            print('Downloading sources...')
        else:
            start = 1
            print(f'Downloading {allMatches} sources...')

        # iterate over all pages in results
        for pageNum in range(start, pages + 1):
            print(f'Page {pageNum} of {pages}...')
            page_response = api(
                "GET",
                '/api/sources',
                {
                    "group_ids": group_ids,
                    'numPerPage': NUM_PER_PAGE,
                    'pageNumber': pageNum,
                    'includeLabellers': True,
                },  # page numbers start at 1
            )
            page_data = page_response.json().get('data')

            for src in page_data['sources']:
                dct = organize_source_data(src)
                dict_list += [dct]

            # create dataframe from query results
            sources = pd.json_normalize(dict_list)
            sources.attrs['scope_code_version'] = code_version
            sources.attrs['fritz_download_dateTime_utc'] = start_dt

            if output_format == '.csv':
                sources.to_csv(filepath, index=False)
            elif output_format == '.h5':
                write_hdf(sources, filepath)
            else:
                write_parquet(sources, filepath)

            print(f'Saved page {pageNum}.')

        if merge_features:
            sources = merge_sources_features(
                sources,
                features_catalog,
                features_limit,
                taxonomy_map,
                output_dir,
                output_filename,
                output_format,
                get_ztf_filters,
                impute_missing_features,
                update_training_set,
                updated_training_set_prefix,
                min_net_votes,
            )

    else:
        # read in CSV, HDF5 or parquet file
        if file.endswith('.csv'):
            sources = pd.read_csv(file)
        elif file.endswith('.h5'):
            sources = read_hdf(file)
        elif file.endswith('.parquet'):
            sources = read_parquet(file)
        else:
            raise TypeError('Input file must be h5, csv or parquet format.')

        if start != 0:
            # continue from checkpoint
            sources = sources[start:]
            filename = (
                os.path.splitext(filename)[0]
                + f'_continued_index_{start}'
                + output_format
            )
            filepath = os.path.join(outpath, filename)

        columns = sources.columns

        missing_col_count = 0
        for colname in [
            'classification',
            'probability',
            'period_origin',
            'period',
        ]:
            if colname not in columns:
                missing_col_count += 1
        if missing_col_count > 0:

            # add obj_id column if not passed in
            if 'obj_id' in columns:
                search_by_obj_id = True
                no_obj_id = 0
            else:
                sources["obj_id"] = None
                search_by_obj_id = False
                no_obj_id = 1

            if ('ra' in columns) & ('dec' in columns):
                racol = sources['ra']
                decol = sources['dec']
                sources.drop(['ra', 'dec'], axis=1, inplace=True)
                no_ra_dec = 0

            if no_obj_id:
                if no_ra_dec:
                    raise KeyError(
                        'Please provide either obj_id or ra and dec columns.'
                    )

            dct_list = []
            for index, row in sources.iterrows():
                # query objects, starting with obj_id
                data = []
                if search_by_obj_id:
                    obj_id = row.obj_id
                    response = api(
                        "GET", '/api/sources/%s' % obj_id, {'includeLabellers': True}
                    )
                    data = response.json().get("data")
                    if len(data) == 0:
                        warnings.warn(
                            'No results from obj_id search - querying by ra/dec.'
                        )
                    else:
                        src = data

                # continue with coordinate search if obj_id unsuccsessful
                if len(data) == 0:
                    # query by ra/dec to get object id
                    ra, dec = racol.loc[index], decol.loc[index]
                    response = api(
                        "GET",
                        f"/api/sources?&ra={ra}&dec={dec}&radius={2/3600}",
                    )
                    data = response.json().get("data")
                    obj_id = None
                    if data["totalMatches"] > 0:
                        src = data["sources"][0]
                        obj_id = src["id"]
                        sources.at[index, 'obj_id'] = obj_id

                print(f"object {index} id:", obj_id)

                # if successful search, get and save labels/probabilities/period annotations to sources dataframe
                if obj_id is not None:
                    dct = organize_source_data(src)
                    dct_list += [dct]

                    # occasional checkpoint at specified number of sources
                    if (index + 1) % CHECKPOINT_NUM == 0:
                        existing_attrs = sources.attrs
                        sources_chkpt = pd.merge(
                            sources, pd.json_normalize(dct_list), on='obj_id'
                        )
                        sources_chkpt.attrs['scope_code_version'] = code_version
                        sources_chkpt.attrs['fritz_download_dateTime_utc'] = start_dt
                        sources.attrs = {**sources.attrs, **existing_attrs}

                        if output_format == '.csv':
                            sources_chkpt.to_csv(filepath, index=False)
                        elif output_format == '.h5':
                            write_hdf(sources_chkpt, filepath)
                        else:
                            write_parquet(sources_chkpt, filepath)

                        print(f'Saved checkpoint at index {index}.')

                else:
                    warnings.warn(f'Unable to find source {index} on Fritz.')

            # final save
            existing_attrs = sources.attrs
            sources = pd.merge(sources, pd.json_normalize(dct_list), on='obj_id')
            sources.attrs['scope_code_version'] = code_version
            sources.attrs['fritz_download_dateTime_utc'] = start_dt
            sources.attrs = {**sources.attrs, **existing_attrs}

            print('Saving all sources.')
            if output_format == '.csv':
                sources.to_csv(filepath, index=False)
            elif output_format == '.h5':
                write_hdf(sources, filepath)
            else:
                write_parquet(sources, filepath)

        if merge_features:
            sources = merge_sources_features(
                sources,
                features_catalog,
                features_limit,
                taxonomy_map,
                output_dir,
                output_filename,
                output_format,
                get_ztf_filters,
                impute_missing_features,
                update_training_set,
                updated_training_set_prefix,
                min_net_votes,
            )

    if update_training_set:

        training_set_extension = pathlib.Path(training_set_path).suffix
        training_set_parent = training_set_path.parent
        training_set_filename = training_set_config.name

        print('Loading training set specified in config...')
        if training_set_extension == '.parquet':
            training_set = read_parquet(training_set_path)
        elif training_set_extension == '.h5':
            training_set = read_hdf(training_set_path)
        elif training_set_extension == '.csv':
            training_set = pd.read_csv(training_set_path)
        else:
            raise ValueError("Training set must be in .parquet, .h5 or .csv format.")

        orig_len = len(training_set)

        updated_training_set = pd.concat([training_set, sources]).reset_index(drop=True)

        dupl_ztf_sources = updated_training_set[
            updated_training_set.duplicated('ztf_id', keep=False)
        ]
        dupl_ztf_ids = dupl_ztf_sources['ztf_id']

        updated_training_set = (
            updated_training_set.set_index('ztf_id').drop(dupl_ztf_ids).reset_index()
        )
        updated_training_set.set_index('obj_id', inplace=True)
        dupl_ztf_sources.set_index('ztf_id', inplace=True)

        # For duplicate sources, use original obj_id and newest classifications
        count_updated = 0
        for u_id in np.unique(dupl_ztf_ids):
            source_set = dupl_ztf_sources.loc[u_id].reset_index()
            first_row = source_set.iloc[0]
            first_oid = first_row['obj_id']
            latest_row = source_set.iloc[-1]
            updated_training_set.loc[first_oid] = latest_row.drop('obj_id')
            count_updated += 1

        updated_training_set = updated_training_set.reset_index()

        updated_filepath = os.path.join(
            BASE_DIR, f"{updated_training_set_prefix}_{training_set_config}"
        )
        updated_filepath = (
            training_set_parent
            / f"{updated_training_set_prefix}_{str(training_set_filename)}"
        )

        if training_set_extension == '.csv':
            updated_training_set.to_csv(updated_filepath, index=False)
        elif training_set_extension == '.h5':
            write_hdf(updated_training_set, updated_filepath)
        else:
            write_parquet(updated_training_set, updated_filepath)

        print(
            f'Saved updated training set containing {len(updated_training_set)} sources ({len(updated_training_set) - orig_len} new, {count_updated} updated).'
        )
        print(
            f'If using scope_upload_classification.py to upload new sources to the existing training set, set --start to {orig_len - count_updated}.'
        )

    return sources


def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default='parse', help="dataset")
    parser.add_argument("--group-ids", type=int, nargs='+', help="list of group ids")
    parser.add_argument(
        "--start", type=int, default=0, help="start page/index for continued download"
    )
    parser.add_argument(
        "--merge-features",
        type=bool,
        nargs='?',
        const=True,
        default=False,
        help="merge downloaded results with features from Kowalski",
    )
    parser.add_argument(
        "--features-catalog",
        type=str,
        default=features_catalog,
        help="catalog of features on Kowalski",
    )

    parser.add_argument(
        "--features-limit",
        type=int,
        default=1000,
        help="Maximum number of sources queried for features per loop",
    )

    parser.add_argument(
        "--taxonomy-map",
        type=str,
        default='golden_dataset_mapper.json',
        help="JSON file mapping between origin labels and Fritz taxonomy",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default='fritzDownload',
        help="Name of directory to save downloaded file",
    )

    parser.add_argument(
        "--output-filename",
        type=str,
        default='merged_classifications_features',
        help="Name of output file containing merged classifications and features",
    )

    parser.add_argument(
        "--output-format",
        type=str,
        default='parquet',
        help="Format of output file: parquet, h5 or csv",
    )

    parser.add_argument(
        "--get-ztf-filters",
        action='store_true',
        default=False,
        help="add ZTF filter ID to default features",
    )

    parser.add_argument(
        "--impute-missing-features",
        action='store_true',
        default=False,
        help="impute missing features using strategy specified by config",
    )

    parser.add_argument(
        "--update-training-set",
        action='store_true',
        default=False,
        help="if downloading an active learning sample, update the training set with the new classification based on votes",
    )

    parser.add_argument(
        "--updated-training-set-prefix",
        type=str,
        default='updated_AL',
        help="Prefix to add to updated trainingset file",
    )

    parser.add_argument(
        "--min-vote-diff",
        type=int,
        default=1,
        help="Minimum number of net votes (upvotes - downvotes) to keep an active learning classification. Caution: if zero, all classifications of reviewed sources will be added",
    )
    return parser


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()

    # download object classifications in the file
    download_classification(
        args.file,
        args.group_ids,
        args.start,
        args.merge_features,
        args.features_catalog,
        args.features_limit,
        args.taxonomy_map,
        args.output_dir,
        args.output_filename,
        args.output_format,
        args.get_ztf_filters,
        args.impute_missing_features,
        args.update_training_set,
        args.updated_training_set_prefix,
        args.min_vote_diff,
    )
