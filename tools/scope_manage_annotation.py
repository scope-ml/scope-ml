#!/usr/bin/env python
import argparse
import pandas as pd
from scope.surveys.fritz import api


def manage_annotation(action, source, group_ids, origin, key, value):

    # Forgiving definitions of actions
    if action in ['update', 'Update', 'UPDATE']:
        action = 'update'
    elif action in ['delete', 'Delete', 'DELETE']:
        action = 'delete'
    elif action in ['post', 'Post', 'POST']:
        action = 'post'

    # Initial checks for origin and key, always necessary
    if origin is None:
        print('Error: please specify origin to %s' % action)
        return

    if key is None:
        print('Error: please specify key to %s' % action)
        return

    # check if source is single object or csv file of many
    if source.endswith('.csv'):
        file = pd.read_csv(source)  # modify input formats to prepare for loop
        if 'obj_id' not in file.columns:
            raise KeyError('CSV file must include column obj_id for ZTF source IDs.')
        obj_ids = file['obj_id']
        if (action == 'update') | (action == 'post'):
            file_values = file[key]
            # Convert numpy dtype to python dtype
            dtype = type(file_values[0].item())
            values = list(map(dtype, file_values.values))

    else:
        # modify single source input formats to prepare for loop
        obj_ids = [source]
        if value is not None:
            if '.' in value:
                values = [float(value)]
            elif value.isdigit():
                values = [int(value)]
            else:
                values = [value]
        else:
            values = [value]

    # loop over objects, performing specified annotation action
    for i in range(len(obj_ids)):
        obj_id = obj_ids[i]

        # update and delete branches require GET for existing annotation
        if (action == 'update') | (action == 'delete'):
            matches = 0
            # get all annotations for object
            response = api("GET", '/api/sources/%s/annotations' % obj_id).json()
            data = response.get('data')

            # loop through annotations, checking for match with input key and origin
            for entry in data:
                annot_id = str(entry['id'])
                annot_origin = entry['origin']
                annot_data = entry['data']

                annot_name = [x for x in annot_data.keys()]
                annot_value = [x for x in annot_data.values()][0]

                for n in annot_name:
                    # if match is found, perform action
                    if origin == annot_origin:
                        if action == 'update':
                            matches += 1
                            value = values[i]

                            # Check value if performing update or post actions
                            if value is None:
                                raise ValueError(
                                    'please specify annotation value to update or post.'
                                )

                            # After passing check, revise annotation with PUT
                            else:
                                annot_data.update({key: value})
                                json = {
                                    "data": annot_data,
                                    "origin": origin,
                                    "obj_id": annot_id,
                                }
                                response = api(
                                    "PUT",
                                    '/api/sources/%s/annotations/%s'
                                    % (obj_id, annot_id),
                                    json,
                                )
                                if response.status_code == 200:  # success
                                    if key == n:
                                        # Updated existing key
                                        print(
                                            'Updated annotation %s (%s = %s to %s) for %s'
                                            % (
                                                annot_origin,
                                                n,
                                                annot_value,
                                                value,
                                                obj_id,
                                            )
                                        )
                                    else:
                                        print(
                                            'Updated annotation %s (%s = %s for %s)'
                                            % (
                                                annot_origin,
                                                key,
                                                value,
                                                obj_id,
                                            )
                                        )
                                    break
                                else:
                                    print('Did not %s - check inputs.' % action)

                        # Delete annotation with DELETE
                        elif action == 'delete':
                            if key == n:
                                matches += 1
                                response = api(
                                    "DELETE",
                                    '/api/sources/%s/annotations/%s'
                                    % (obj_id, annot_id),
                                )
                                if response.status_code == 200:  # success
                                    print(
                                        'Deleted annotation %s (%s = %s) for %s'
                                        % (annot_origin, n, annot_value, obj_id)
                                    )
                                else:
                                    print('Did not %s - check inputs.' % action)

            # Alert user if no origin/key matches in each source's annotations
            if matches == 0:
                print(
                    'Origin and/or key (%s, %s) did not match any existing annotations for %s.'
                    % (origin, key, obj_id)
                )

        # if posting new annotation, skip search for exisiting ones
        elif action == 'post':
            value = values[i]

            # Check value if performing update or post actions
            if value is None:
                raise ValueError('please specify annotation value to update or post.')

            # After passing check, post annotation with POST
            else:
                json = {"origin": origin, "data": {key: value}, "group_ids": group_ids}
                response = api("POST", '/api/sources/%s/annotations' % obj_id, json)
                if response.status_code == 200:  # success
                    print(
                        'Posted annotation %s (%s = %s) for %s'
                        % (origin, key, value, obj_id)
                    )
                else:
                    print(
                        'Did not %s - check inputs and existing annotations.' % action
                    )

        # Must choose one of the three specified actions
        else:
            print(
                "Error: please specify action as one of 'post', 'update', or 'delete'."
            )


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--action", help="post, update, or delete annotation")
    parser.add_argument("--source", help="Fritz object id or csv file of sources")
    parser.add_argument("--group-ids", type=int, nargs='+', help="list of group ids")
    parser.add_argument("--origin", type=str, help="name of annotation origin")
    parser.add_argument("--key", help="annotation key")
    parser.add_argument("--value", type=str, help="annotation value")

    return parser


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()

    manage_annotation(
        args.action,
        args.source,
        args.group_ids,
        args.origin,
        args.key,
        args.value,
    )
