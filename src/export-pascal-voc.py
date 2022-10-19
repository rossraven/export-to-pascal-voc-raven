import os
import lxml.etree as ET
import supervisely as sly
from shutil import copyfile
from supervisely.app.v1.app_service import AppService
from json2xml import json2xml

my_app = AppService()

TEAM_ID = int(os.environ['context.teamId'])
WORKSPACE_ID = int(os.environ['context.workspaceId'])
PROJECT_ID = int(os.environ['modal.state.slyProjectId'])
DATASET_ID = os.environ['modal.state.slyDatasetId']
if DATASET_ID is not None:
    DATASET_ID = int(DATASET_ID)

PASCAL_CONTOUR_THICKNESS = int(os.environ['modal.state.pascalContourThickness'])
TRAIN_VAL_SPLIT_COEF = float(os.environ['modal.state.trainSplitCoef'])

ARCHIVE_NAME_ENDING = '_pascal_voc.tar.gz'
RESULT_DIR_NAME_ENDING = '_pascal_voc'
RESULT_SUBDIR_NAME = 'VOCdevkit/VOC'

images_dir_name = 'Images'
ann_dir_name = 'Annotations'

trainval_sets_dir_name = 'train-val'
trainval_sets_main_name = 'class-split'
trainval_sets_segm_name = 'data-split'

train_txt_name = 'train.txt'
val_txt_name = 'val.txt'

is_trainval = None

TRAIN_TAG_NAME = 'train'
VAL_TAG_NAME = 'val'
SPLIT_TAGS = set([TRAIN_TAG_NAME, VAL_TAG_NAME])

VALID_IMG_EXT = set(['.jpe', '.jpeg', '.jpg'])
SUPPORTED_GEOMETRY_TYPES = set([sly.Bitmap, sly.Polygon, sly.Rectangle])

if TRAIN_VAL_SPLIT_COEF > 1 or TRAIN_VAL_SPLIT_COEF < 0:
    raise ValueError('train_val_split_coef should be between 0 and 1, your data is {}'.format(TRAIN_VAL_SPLIT_COEF))

def ann_to_xml(project_info, image_info, img_filename, result_ann_dir, ann):
    xml_root = ET.Element("annotation")

    ET.SubElement(xml_root, "filename").text = img_filename

    xml_root_source = ET.SubElement(xml_root, "source")
    ET.SubElement(xml_root_source, "database").text = "Supervisely Project ID:" + str(project_info.id)
    ET.SubElement(xml_root_source, "annotation").text = "PASCAL VOC"
    ET.SubElement(xml_root_source, "image").text = "Supervisely Image ID:" + str(image_info.id)

    xml_root_size = ET.SubElement(xml_root, "size")
    ET.SubElement(xml_root_size, "width").text = str(image_info.width)
    ET.SubElement(xml_root_size, "height").text = str(image_info.height)
    ET.SubElement(xml_root_size, "depth").text = "3"

    ET.SubElement(xml_root, "segmented").text = "1" if len(ann.labels) > 0 else "0"

    for label in ann.labels:
        if label.obj_class.name == "neutral":
            continue

        bitmap_to_bbox = label.geometry.to_bbox()

        xml_ann_obj = ET.SubElement(xml_root, "object")
        ET.SubElement(xml_ann_obj, "name").text = label.obj_class.name
        ET.SubElement(xml_ann_obj, "pose").text = "Unspecified"
        ET.SubElement(xml_ann_obj, "truncated").text = "0"
        ET.SubElement(xml_ann_obj, "difficult").text = "0"

        xml_ann_obj_bndbox = ET.SubElement(xml_ann_obj, "bndbox")
        ET.SubElement(xml_ann_obj_bndbox, "xmin").text = str(bitmap_to_bbox.left)
        ET.SubElement(xml_ann_obj_bndbox, "ymin").text = str(bitmap_to_bbox.top)
        ET.SubElement(xml_ann_obj_bndbox, "xmax").text = str(bitmap_to_bbox.right)
        ET.SubElement(xml_ann_obj_bndbox, "ymax").text = str(bitmap_to_bbox.bottom)

    tree = ET.ElementTree(xml_root)

    img_name = os.path.join(result_ann_dir, img_filename + ".xml")
    ann_path = (os.path.join(result_ann_dir, img_name))
    ET.indent(tree, space="    ")
    tree.write(ann_path, pretty_print=True)


def find_first_tag(img_tags, split_tags):
    for tag in split_tags:
        if img_tags.has_key(tag):
            return img_tags.get(tag)
    return None


def write_main_set(is_trainval, images_stats, meta_json, result_imgsets_dir):
    result_imgsets_main_subdir = os.path.join(result_imgsets_dir, trainval_sets_main_name)
    result_imgsets_segm_subdir = os.path.join(result_imgsets_dir, trainval_sets_segm_name)
    sly.fs.mkdir(result_imgsets_main_subdir)

    res_files = ["trainval.txt", "train.txt", "val.txt"]
    for file in os.listdir(result_imgsets_segm_subdir):
        if file in res_files:
           copyfile(os.path.join(result_imgsets_segm_subdir, file), os.path.join(result_imgsets_main_subdir, file))

    train_imgs = [i for i in images_stats if i['dataset'] == TRAIN_TAG_NAME]
    val_imgs = [i for i in images_stats if i['dataset'] == VAL_TAG_NAME]

    write_objs = [
        {'suffix': 'trainval', 'imgs': images_stats},
        {'suffix': 'train', 'imgs': train_imgs},
        {'suffix': 'val', 'imgs': val_imgs},
    ]

    if is_trainval == 1:
       trainval_imgs = [i for i in images_stats if i['dataset'] == TRAIN_TAG_NAME + VAL_TAG_NAME]
       write_objs[0] =  {'suffix': 'trainval', 'imgs': trainval_imgs}

    for obj_cls in meta_json.obj_classes:
        if obj_cls.geometry_type not in SUPPORTED_GEOMETRY_TYPES:
            continue
        if obj_cls.name == 'neutral':
            continue
        for o in write_objs:
            with open(os.path.join(result_imgsets_main_subdir, f'{obj_cls.name}_{o["suffix"]}.txt'), 'w') as f:
                for img_stats in o['imgs']:
                    v = "1" if obj_cls.name in img_stats['classes'] else "-1"
                    f.write(f'{img_stats["name"]} {v}\n')


def write_segm_set(is_trainval, images_stats, result_imgsets_dir):
    result_imgsets_segm_subdir = os.path.join(result_imgsets_dir, trainval_sets_segm_name)
    sly.fs.mkdir(result_imgsets_segm_subdir)

    with open(os.path.join(result_imgsets_segm_subdir, 'trainval.txt'), 'w') as f:
        if is_trainval ==1:
            f.writelines(i['name'] + '\n' for i in images_stats if i['dataset'] == TRAIN_TAG_NAME+VAL_TAG_NAME)
        else:
            f.writelines(i['name'] + '\n' for i in images_stats)
    with open(os.path.join(result_imgsets_segm_subdir, 'train.txt'), 'w') as f:
        f.writelines(i['name'] + '\n' for i in images_stats if i['dataset'] == TRAIN_TAG_NAME)
    with open(os.path.join(result_imgsets_segm_subdir, 'val.txt'), 'w') as f:
        f.writelines(i['name'] + '\n' for i in images_stats if i['dataset'] == VAL_TAG_NAME)


@my_app.callback("from_sly_to_pascal")
@sly.timeit
def from_sly_to_pascal(api: sly.Api, task_id, context, state, app_logger):
    global PASCAL_CONTOUR_THICKNESS, TRAIN_VAL_SPLIT_COEF

    project_info = api.project.get_info_by_id(PROJECT_ID)
    meta_json = api.project.get_meta(PROJECT_ID)
    meta = sly.ProjectMeta.from_json(meta_json)
    app_logger.info("Palette has been created")

    full_archive_name = str(project_info.id) + '_' + project_info.name + ARCHIVE_NAME_ENDING
    full_result_dir_name = str(project_info.id) + '_' + project_info.name + RESULT_DIR_NAME_ENDING

    result_archive = os.path.join(my_app.data_dir, full_archive_name)
    result_dir = os.path.join(my_app.data_dir, full_result_dir_name)
    result_subdir = os.path.join(result_dir, RESULT_SUBDIR_NAME)

    result_ann_dir = os.path.join(result_subdir, ann_dir_name)
    result_images_dir = os.path.join(result_subdir, images_dir_name)
    result_imgsets_dir = os.path.join(result_subdir, trainval_sets_dir_name)

    sly.fs.mkdir(result_ann_dir)
    sly.fs.mkdir(result_imgsets_dir)
    sly.fs.mkdir(result_images_dir)

    app_logger.info("Pascal VOC directories have been created")

    images_stats = []
    count = 0

    if DATASET_ID is not None:
        dataset_info = api.dataset.get_info_by_id(DATASET_ID)
        datasets = [dataset_info]
    else:
        datasets = api.dataset.get_list(PROJECT_ID)
    
    for ds in datasets:
        count += ds.images_count
    
    dataset_names = ['trainval', 'val', 'train']
    progress = sly.Progress('Preparing images for export', count, app_logger)
    for dataset in datasets:
        if dataset.name in dataset_names:
           is_trainval = 1
        else:
           is_trainval = 0

        images = api.image.get_list(dataset.id)
        for batch in sly.batched(images):
            image_ids = [image_info.id for image_info in batch]
            image_paths = [os.path.join(result_images_dir, image_info.name) for image_info in batch]

            api.image.download_paths(dataset.id, image_ids, image_paths)
            ann_infos = api.annotation.download_batch(dataset.id, image_ids,with_custom_data=True)
            for image_info, ann_info in zip(batch, ann_infos):
                img_title, img_ext = os.path.splitext(image_info.name)
                cur_img_filename = image_info.name

                if is_trainval == 1:
                    cur_img_stats = {'classes': set(), 'dataset': dataset.name, 'name': img_title}
                    images_stats.append(cur_img_stats)
                else:
                    cur_img_stats = {'classes': set(), 'dataset': None, 'name': img_title}
                    images_stats.append(cur_img_stats)

                if img_ext not in VALID_IMG_EXT:
                    orig_image_path = os.path.join(result_images_dir, cur_img_filename)

                    jpg_image = img_title + ".jpg"
                    jpg_image_path = os.path.join(result_images_dir, jpg_image)

                    im = sly.image.read(orig_image_path)
                    sly.image.write(jpg_image_path, im)
                    sly.fs.silent_remove(orig_image_path)
                
                if ann_info.annotation['customBigData'] is not None:
                    data = ann_info.annotation['customBigData']
                    xml_ann = json2xml.Json2xml(data,root=False,item_wrap=False,attr_type=False).to_xml()
                    img_name = os.path.join(result_ann_dir, cur_img_filename + ".xml")
                    ann_path = (os.path.join(result_ann_dir, img_name))
                    with open(ann_path,'w') as f: f.write(xml_ann)
                    continue
                else:
                    ann_info.annotation['customBigData'] = {}

                ann = sly.Annotation.from_json(ann_info.annotation, meta)
                tag = find_first_tag(ann.img_tags, SPLIT_TAGS)
                if tag is not None:
                    cur_img_stats['dataset'] = tag.meta.name

                valid_labels = []
                for label in ann.labels:
                    if type(label.geometry) in SUPPORTED_GEOMETRY_TYPES:
                        valid_labels.append(label)
                    else:
                        app_logger.warn(
                            f"Label has unsupported geometry type ({type(label.geometry)}) and will be skipped.")

                ann = ann.clone(labels=valid_labels)
                ann_to_xml(project_info, image_info, cur_img_filename, result_ann_dir, ann)
                for label in ann.labels:
                    cur_img_stats['classes'].add(label.obj_class.name)

                progress.iter_done_report()

    imgs_to_split = [i for i in images_stats if i['dataset'] is None]
    train_len = int(len(imgs_to_split) * TRAIN_VAL_SPLIT_COEF)

    for img_stat in imgs_to_split[:train_len]: img_stat['dataset'] = TRAIN_TAG_NAME
    for img_stat in imgs_to_split[train_len:]: img_stat['dataset'] = VAL_TAG_NAME

    write_segm_set(is_trainval, images_stats, result_imgsets_dir)
    write_main_set(is_trainval, images_stats, meta, result_imgsets_dir)

    sly.fs.archive_directory(result_dir, result_archive)
    app_logger.info("Result directory is archived")

    upload_progress = []
    remote_archive_path = "/ApplicationsData/Export-to-Pascal-VOC/{}/{}".format(task_id, full_archive_name)

    def _print_progress(monitor, upload_progress):
        if len(upload_progress) == 0:
            upload_progress.append(sly.Progress(message="Upload {!r}".format(full_archive_name),
                                                total_cnt=monitor.len,
                                                ext_logger=app_logger,
                                                is_size=True))
        upload_progress[0].set_current_value(monitor.bytes_read)

    file_info = api.file.upload(TEAM_ID, result_archive, remote_archive_path,
                                lambda m: _print_progress(m, upload_progress))
    app_logger.info("Uploaded to Team-Files: {!r}".format(file_info.storage_path))
    api.task.set_output_archive(task_id, file_info.id, full_archive_name, file_url=file_info.storage_path)

    my_app.stop()


def main():
    sly.logger.info("Script arguments", extra={
        "TEAM_ID": TEAM_ID,
        "WORKSPACE_ID": WORKSPACE_ID,
        "PROJECT_ID": PROJECT_ID,
        "DATASET_ID": DATASET_ID
    })
    
    my_app.run(initial_events=[{"command": "from_sly_to_pascal"}])

if __name__ == '__main__':
    sly.main_wrapper("main", main)
