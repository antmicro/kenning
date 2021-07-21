"""
A script that uses the model in inference-only mode with
no performance data collection. It utilizes a Dataprovider and
one or more OutputCollectors to supply and gather data to and from the model
and save them or visualize to the user.

It requires implementations of several classes as input:

* Dataprovider : provides data for inference
* Outputcollector : interprets and visualizes the output
* Runtime : runs the model with provided data and returns the output

Each class requires arguments to configure them and provide user settings
"""
import sys
import argparse

from kenning.utils.class_loader import load_class
# import kenning.utils.logger as logger
import cv2


def main(argv):
    parser = argparse.ArgumentParser(argv[0], add_help=False)
    parser.add_argument(
        'modelwrappercls',
        help='ModelWrapper-based class with inference implementation to import',  # noqa: E501
    )
    parser.add_argument(
        'runtimecls',
        help='Runtime-based class with the implementation of model runtime'
    )
    parser.add_argument(
        'dataprovidercls',
        help='Dataprovider-based class used for providing data',
    )
    parser.add_argument(
        'outputcollectorcls',
        help='Outputcollector-based class for visualizing and gathering data',
        action='append'
    )

    args, _ = parser.parse_known_args(argv[1:])

    modelwrappercls = load_class(args.modelwrappercls)
    runtimecls = load_class(args.runtimecls)
    dataprovidercls = load_class(args.dataprovidercls)
    outputcollectorcls = []

    for i in args.outputcollectorcls:
        outputcollectorcls.append(load_class(i))

    parser = argparse.ArgumentParser(
        argv[0],
        parents=[
            parser,
            modelwrappercls.form_argparse()[0],
            runtimecls.form_argparse()[0],
            dataprovidercls.form_argparse()[0]
        ] + ([i.form_argparse()[0] for i in outputcollectorcls])
    )

    args = parser.parse_args(argv[1:])

    dataprovider = dataprovidercls.from_argparse(args)
    model = modelwrappercls.from_argparse(dataprovider, args)
    runtime = runtimecls.from_argparse(None, args)
    outputcollectors = [
        o.from_argparse(args) for o in outputcollectorcls
    ]

    runtime.prepare_model(None)
    keycode = 0
    try:
        while keycode != 27:
            img_inp = dataprovider.get_input()
            inp = model.convert_input_to_bytes(model.preprocess_input(img_inp))
            runtime.prepare_input(inp)
            runtime.run()
            res = model.postprocess_outputs(
                model.convert_output_from_bytes(runtime.upload_output(inp))
            )
            for i in outputcollectors:
                i.return_output(img_inp, res)
            keycode = cv2.waitKey(1)
    except KeyboardInterrupt:
        log.info("Interrupt signal caught, shutting down (press CTRL-C again to force quit)")  # noqa E501
        dataprovider.detach_from_source()
        for o in outputcollectors:
            o.detach_from_output()


if __name__ == '__main__':
    main(sys.argv)
