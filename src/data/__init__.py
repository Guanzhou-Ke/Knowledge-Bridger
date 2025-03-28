from .datasets import COCOMultiLabelDataset, MMIMDBMultiLabelDataset, IUXRAYMultiLabelDataset
from pathlib import Path

def get_dataset(name, args):
    if name == 'coco-classificaiton':
        train_dataset = COCOMultiLabelDataset(Path(args['train_root']),
                                              clip_download_root=Path(args['clip_download_root']),
                                              class_name_path=Path(args['class_name_path']),
                                              metadata_path=Path(args['train_metadata_path']),
                                              isMissing=args['is_missing'],
                                              missing_config=Path(args['missing_config']),
                                              keep_raw=args['keep_raw'],
                                              merge_metadata=args['merge_metadata']
                                              )

        test_dataset = COCOMultiLabelDataset(Path(args['val_root']),
                                             clip_download_root=Path(args['clip_download_root']),
                                             class_name_path=Path(args['class_name_path']),
                                             metadata_path=Path(args['test_metadata_path']),
                                             )
    elif name == 'mmimdb':
        train_dataset = MMIMDBMultiLabelDataset(Path(args['train_root']),
                                              clip_download_root=Path(args['clip_download_root']),
                                              class_name_path=Path(args['class_name_path']),
                                              metadata_path=Path(args['train_metadata_path']),
                                              isMissing=args['is_missing'],
                                              missing_config=Path(args['missing_config']),
                                              keep_raw=args['keep_raw'],
                                              merge_metadata=args['merge_metadata']
                                              )

        test_dataset = MMIMDBMultiLabelDataset(Path(args['val_root']),
                                             clip_download_root=Path(args['clip_download_root']),
                                             class_name_path=Path(args['class_name_path']),
                                             metadata_path=Path(args['test_metadata_path']),
                                             )
    elif name == 'iuxray':
        train_dataset = IUXRAYMultiLabelDataset(Path(args['train_root']),
                                              clip_download_root=Path(args['clip_download_root']),
                                              class_name_path=Path(args['class_name_path']),
                                              metadata_path=Path(args['train_metadata_path']),
                                              isMissing=args['is_missing'],
                                              missing_config=Path(args['missing_config']),
                                              keep_raw=args['keep_raw'],
                                              merge_metadata=args['merge_metadata']
                                              )

        test_dataset = IUXRAYMultiLabelDataset(Path(args['val_root']),
                                             clip_download_root=Path(args['clip_download_root']),
                                             class_name_path=Path(args['class_name_path']),
                                             metadata_path=Path(args['test_metadata_path']),
                                             )
    else:
        raise ValueError(f"Dataset {name} is not supported.")
    
    return train_dataset, test_dataset
